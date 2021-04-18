
using JSON, DataFrames, JSON3, JSONTables
using LinearAlgebra
using JLD2
using StatsBase
using MLBase
using GLM
using Lathe.preprocess: TrainTestSplit

competition_id=43;
season_id=3;

function get_match_ids(competition_id, season_id)
    data = JSON.parsefile("./football_data/statsbomb/data/matches/$competition_id/$season_id.json")
    data = reduce(vcat, DataFrame.(data), cols=:union)
    return data[:,"match_id"]
end


function get_events(match_ids)
    events = []
    df=0
    base_path = "./football_data/statsbomb/data/events/"
    for id in match_ids
        path = base_path * "$id" * ".json"
        str = read(path, String)
        jtable = jsontable(JSON3.read(str))
        # append!(events, DataFrame(jtable))
        if df == 0
            df = DataFrame(jtable)
        else
            df2 = DataFrame(jtable)
            append!(df, df2, cols=:union)
        end
    end
    return df
end

function filter_events(events, event_id)
    return filter(:type => type-> type.id == event_id, events)
end

function shot_distance(origin)
    """ Return the distance between goal and origin"""
    goal = [120.0,40.0]
    return sqrt(sum((origin - goal) .^ 2))
end

function shot_angle(origin)
    """ Return the view angle the origin can see of the goal"""
    p0 = [120., 36.]  # Left Post
    p1 = [120., 44.]  # Right Post

    v0 = p0 - origin
    v1 = p1 - origin

    angle = abs(atan(det(reshape([v0...;v1...], 2, 2)), dot(v0, v1)))

    return angle
end

function add_shot_data(shots)
    """ Return triple of origin data, destination data, outcome.
    """
    N = size(shots)[1]
    p2ps = zeros(N,2)
    distances = zeros(N)
    angles = zeros(N)
    outcomes = zeros(N)

    for i in 1:N
        origin = copy(shots[i,"location"])
        outcome = "Goal" == shots[i,"shot"]["outcome"]["name"]
        p2ps[i,1] = origin[1]
        p2ps[i,2] = origin[2]
        distances[i] = shot_distance(origin)
        angles[i] = shot_angle(origin)
        outcomes[i] = outcome
    end
    insertcols!(shots, 1, :x_bin =>  p2ps[:,1])
    insertcols!(shots, 1, :y_bin =>  p2ps[:,2])
    insertcols!(shots, 1, :outcome => outcomes)
    insertcols!(shots, 1, :distance => distances)
    insertcols!(shots, 1, :angle => angles)
end

function filter_shot_data(shots)
    select!(shots, [:angle, :distance, :outcome, :x_bin, :y_bin])
end

function normalize_shots(shots, cols)
    for col in cols
        max_datum = maximum(shots[:, col])
        min_datum = minimum(shots[:, col])
        DataFrames.transform!(shots, col =>  data -> ((data .- min_datum) ./ (max_datum - min_datum)))
    end
end

function sample_all_shots(df)
    all_locations = reverse.(Iterators.product(0:80,60:120))[:]
    N = size(all_locations)[1]
    p2ps = zeros(N,2)
    distances = zeros(N)
    angles = zeros(N)
    outcomes = zeros(N)

    for i in 1:N
        origin = [all_locations[i][1],all_locations[i][2]]
        p2ps[i,1] = origin[1]
        p2ps[i,2] = origin[2]
        distances[i] = shot_distance(origin)
        angles[i] = shot_angle(origin)
    end
    insertcols!(df, 1, :x_bin =>  p2ps[:,1])
    insertcols!(df, 1, :y_bin =>  p2ps[:,2])
    insertcols!(df, 1, :distance => distances)
    insertcols!(df, 1, :angle => angles)
end

ids = get_match_ids(competition_id, season_id);
events = get_events(ids);
shots = filter_events(events, 16) ### Shots have id=16

add_shot_data(shots)
@info "type: $(typeof(shots)) + size: $(size(shots))"

filter_shot_data(shots)
@info "type: $(typeof(shots)) + size: $(size(shots))"
@info "description: $(describe(shots))"

normalize_shots(shots, [:x_bin, :y_bin, :distance, :angle])
@info "type: $(typeof(shots)) + size: $(size(shots))"
@info "description: $(describe(shots))"

train, test = TrainTestSplit(shots,.75);
logits = []
formulas = [@formula(outcome ~ x_bin_function + y_bin_function),
            @formula(outcome ~ distance_function),
            @formula(outcome ~ angle_function),
            @formula(outcome ~ x_bin_function + y_bin_function + distance_function + angle_function)]

for formula in formulas
    push!(logits,glm(formula, train, Binomial(), ProbitLink()))
end
names = ["location", "angle", "distance", "location, distance, and angle"]

for i in 1:size(logits)[1]
    @info "Logistic Regression based on $(names[i])"
    local prediction = predict(logits[i],test)
    local prediction_class = [if x < 0.5 0 else 1 end for x in prediction];
    local prediction_df = DataFrame(outcome = test.outcome, prediction = prediction_class, probability = prediction);
    local prediction_df.correctly_classified = prediction_df.outcome .== prediction_df.prediction
    local accuracy = mean(prediction_df.correctly_classified)
    @info "\t accuracy: $(accuracy)"
end

df = DataFrame()
sample_all_shots(df)
normalize_shots(df, [:x_bin, :y_bin, :distance, :angle])
@info "type: $(typeof(df)) + size: $(size(df))"
@info "$(describe(df))"
predictions = []
for logit in logits
    local prediction = predict(logit,df)
    push!(predictions, reshape(prediction,(81,61)))
end

using Plots, Images

function draw_pitch!()
    # Sidelines
    plot!(Shape([(0,0), (60,0), (60,80), (0,80)]), fillcolor = nothing, color=:black, label=nothing)

    # 18yd box
    plot!(Shape([(60,18), (42,18), (42,62), (60, 62)]), fillcolor = nothing, color=:black, label=nothing)

    # 10yd box
    plot!(Shape([(60,30), (54,30), (54,50), (60, 50)]), fillcolor = nothing, color=:black, label=nothing)

    # Half line
    plot!(Shape([(60,0), (60,80)]), fillcolor = nothing, color=:black, label=nothing)
end


begin
    plots = []
    for prediction in predictions
        local pred_surface = imfilter(prediction, Kernel.gaussian(0.5))
        local p = plot(aspect_ratio=:equal,axis=nothing, framestyle=:none, border=:none)
        Plots.heatmap!(pred_surface, xflip=true, fill=true, seriescolor=cgrad(:summer,scale=:exp), legendfontsize=9)
        draw_pitch!()
        push!(plots, p)
    end

    display(plot(plots[1], framestyle=:box))
    display(plot(plots[2], plots[3], layout=2))
    display(plot(plots[4], framestyle=:box))
end