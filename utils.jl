using JSON, DataFrames, JSON3, JSONTables
using LinearAlgebra: norm
using JLD2

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


function point_to_pitch(point; x_bins=52, y_bins=34)
    """ Return a matrix of size (x_bins, y_bins) where the only non-zero entry is at the bin corresponding to point's coordinates
    """
    pitch = zeros((x_bins, y_bins))
    xs=LinRange(0, 120, x_bins)
    ys=LinRange(0, 80, y_bins)
    x,y = point
    x_i = searchsortedfirst(xs, x)
    y_i = searchsortedfirst(ys, y)
    pitch[x_i, y_i] = 1.
    return pitch
end


function pitch_distance(grid_point, origin)
    """ Return the distance between grid_point and origin"""
    return norm(grid_point .- origin)
end


function tensor(point; x_bins=52, y_bins=34)
    """ Return a 52x34x3 array that corresponds to the input data for the coordinates point.
    """
    x = range(0,stop=120,length=x_bins)
    y = range(0,stop=80,length=y_bins)
    z_grid = Iterators.product(x,y) # meshgrid for contour
    z_grid = reshape.(collect.(z_grid),:,1) # add single batch dim

    goal=[120;40]
    f(grid_point) = pitch_distance(grid_point, point)
    g(grid_point) = pitch_distance(grid_point, goal)

    pos = point_to_pitch(point)
    d_origin = f.(z_grid)
    d_goal = g.(z_grid)

    return reshape(cat(pos, d_origin, d_goal, dims=2), (x_bins, y_bins, 3))
end


function get_data(passes; x_bins=52, y_bins=34)
    """ Return triple of origin data, destination data, outcome.
    DIMENSIONS:
        - origin: (x_bins, y_bins, 3, N)
        - destination: (x_bins, y_bins, 1, N)
        - outcome: (N,)
    """
    N = size(passes)[1]
    dataXp = zeros((x_bins, y_bins, 3, N));
    dataXd = zeros((x_bins, y_bins, 1, N));
    dataY = zeros(N)

    for i in 1:N
        origin = copy(passes[i,"location"])
        outcome = !("outcome" in keys(passes[i,"pass"]))
        destination = copy(passes[i,"pass"]["end_location"])
    
        dataXp[:,:,:,i] = tensor(origin)
        dataXd[:,:,:,i] = point_to_pitch(destination)
        dataY[i] = outcome
    end

    return dataXp, dataXd, dataY
end


ids = get_match_ids(competition_id, season_id);
events = get_events(ids);
passes = filter_events(events, 30) ### Passes have id=30
Xp, Xd, Y = get_data(passes)

Xp = convert(Array{Float32, 4}, Xp)
Xd = convert(Array{Float32, 4}, Xd)
Y = convert(Vector{Float32}, Y)

@save "dataset.jld2" {compress=true} data=(Xp, Xd, Y)
