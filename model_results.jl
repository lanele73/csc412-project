using JLD2, BSON, Flux, Zygote, Plots, Images, Random
using Statistics: mean


##### Load model #####

function unpack(path)
    @load path data
    return data[1], data[2], data[3]
end

begin
    Xp, Xd, Y = unpack("dataset.jld2")
    shuffle_index = shuffle(MersenneTwister(1), 1:size(Xp)[4])

    Xp = Xp[:,:,:,shuffle_index]
    Xd = Xd[:,:,:,shuffle_index]
    Y = Y[shuffle_index]

    split_index = Int(floor(size(Y)[1] * 0.9))

    Xp_train = Xp[:,:,:,1:split_index]
    Xd_train = Xd[:,:,:,1:split_index]
    Y_train = Y[1:split_index]

    Xp_test = Xp[:,:,:,split_index+1:end]
    Xd_test = Xd[:,:,:,split_index+1:end]
    Y_test = Y[split_index+1:end]
end


function const_init(d1, d2, d3, d4)
    val = mean(Y_train)
    return val .* ones(Float32,(d1, d2, d3, d4))
end

conv_net = Chain(
    Conv((3, 3), 3=>16, relu),          # No padding
    symm_pad,                           # Symmetric padding
    Conv((1, 1), 16=>1),                

    MaxPool((2, 2), pad=SamePad()),     # Same padding
    Conv((3, 3), 1=>32, relu),
    symm_pad,
    Conv((1, 1), 32=>1),                # Symmetric padding

    Upsample((2,2)),
    Conv((3,3), 1=>16, relu),           # No padding
    symm_pad,                           # Symmetric padding  
    Conv((1,1), 16=>1, sigmoid;         
        init=const_init)
)

function pixel_layer(x)
    surface = x[:,:,:,:,1]
    mask = x[:,:,:,:,2]
    masked = surface .* mask
    return sum(masked, dims=(2,1))
end

function loss(batch_Xp, batch_Xd, Y)
    N = size(Y)[1]
    NN_out = conv_net(batch_Xp)
    x = cat(NN_out, batch_Xd, dims=5)
    pixel = reshape(pixel_layer(x), (N,))
    return sum(Flux.Losses.binarycrossentropy.(pixel, Y))/N
end

bs = 2000
batches = Flux.Data.DataLoader((Xp_train,Xd_train,Y_train); batchsize=bs, shuffle=false)
num_batches = Int(ceil(size(Y_train)[1] / bs))

load_model = true
if load_model
    using Zygote
    version = 25
    BSON.@load "saved_runs/params$(version).bson" ps
    BSON.@load "saved_runs/loss_history$(version).bson" history
    batch_loss_values, epoch_loss_values = history
    Flux.loadparams!(conv_net, ps)
    batch_Xp = first(batches)[1]
    batch_Xd = first(batches)[2]
    Y = first(batches)[3]
    loss(batch_Xp, batch_Xd, Y)  ### Loss should be around 0.38
end


##### Helper functions #####

function draw_pitch!()
    # Sidelines
    plot!(Shape([(0,0), (120,0), (120,80), (0,80)]), fillcolor = nothing, color=:black, label=nothing)         

    # 18yd box
    plot!(Shape([(0,18), (18,18), (18,62), (0, 62)]), fillcolor = nothing, color=:black, label=nothing)
    plot!(Shape([(120,18), (102,18), (102,62), (120, 62)]), fillcolor = nothing, color=:black, label=nothing)

    # 10yd box
    plot!(Shape([(0,30), (6,30), (6,50), (0, 50)]), fillcolor = nothing, color=:black, label=nothing)
    plot!(Shape([(120,30), (114,30), (114,50), (120, 50)]), fillcolor = nothing, color=:black, label=nothing)

    # Half line
    plot!(Shape([(60,0), (60,80)]), fillcolor = nothing, color=:black, label=nothing)
end


function draw_probs(sample_input)
    pred_surface = imfilter(conv_net(sample_input)[:,:,1,1], Kernel.gaussian(0.5))'
    xs=LinRange(0,120,52)
    ys=LinRange(0,80,34)
    p = plot(aspect_ratio=:equal,axis=nothing,border=:none)
    contour!(xs, ys, pred_surface, fill=true, seriescolor=:summer, linecolor=:black, linewidth=0.5, colorbar=false)
    x = findmax(sample_input[:,:,1])[2][1] - 1
    y = findmax(sample_input[:,:,1])[2][2] - 1
    scatter!([x*120/52],[y*80/34], color=:red, markersize=5, label=nothing)
    draw_pitch!()
    return p, surf
end

### Look at samples
begin
    sample_index = rand(MersenneTwister(1), (1:size(Y_test)[1]), 4)
    sample_input = Xp_test[:,:,:,sample_index]

    plots = []
    surfaces = []
    for i in 1:4
        p, surf = draw_probs(sample_input[:,:,:,i:i])
        push!(surfaces, surf)
        push!(plots, p)
    end
    clim = (minimum(minimum.(surfaces)), maximum(maximum.(surfaces)))
    h2 = scatter([0,0], [0,1], zcolor=[0,1], xlims=(1,1.1), label="",clims=clim, c=:summer, framestyle=:none)
    l = @layout [grid(2, 2) a{0.035w}]
    plot(plots..., h2, layout=l)
end