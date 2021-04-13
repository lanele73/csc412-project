using JLD2, Flux, Random, Plots
using MLDataPattern: splitobs
using Statistics: mean


function unpack(path)
    @load path data
    return data[1], data[2], data[2]
end


### Load Data ###
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

    # Xp_train, Xp_test = splitobs(Xp, at=0.9)
    # Xd_train, Xd_test = splitobs(Xd, at=0.9)
    # Y_train, Y_test = splitobs(Y, at=0.9)
end


function const_init(d1, d2, d3, d4)
    val = mean(Y_train)
    return val .* ones(Float32,(d1, d2, d3, d4))
end


# let
#     p1=heatmap(Xp_train[:,:,1,2])
#     p2=heatmap(Xp_train[:,:,2,2])
#     p3=heatmap(Xp_train[:,:,3,2])
#     plot(p1, p2, p3, layout=3)
# end


pass_net = Chain(
    Conv((3, 3), 3=>16, relu),              # No padding
    Conv((1, 1), 16=>1; pad=(1,1)),         # Symmetric padding

    MaxPool((2, 2), pad=SamePad()),         # Same padding
    Conv((3, 3), 1=>32, relu),
    Conv((1, 1), 32=>1; pad=(1,1)),         # Symmetric padding

    Upsample((2,2)),
    Conv((3,3), 1=>16, relu),               # No padding
    Conv((1,1), 16=>1, sigmoid;             # Symmetric padding
        pad=(1,1), init=const_init)
)

# pass_net = f64(pass_net)

function pixel_layer(x)
    surface = x[:,:,:,:,1]
    mask = x[:,:,:,:,2]
    masked = surface .* mask
    return sum(masked, dims=(2,1))
end


function loss(batch_Xp, batch_Xd, Y)
    N = size(Y)[1]
    NN_out = pass_net(batch_Xp)
    x = cat(NN_out, batch_Xd, dims=5)
    pixel = reshape(pixel_layer(x), size(Y))
    return sum(Flux.Losses.logitbinarycrossentropy.(pixel, Y))/N
end


begin
    batch_Xp = Xp_test[:,:,:,1:23]
    batch_Xd = Xd_test[:,:,:,1:23]
    outcome = Y_test[1:23]
    ps = Flux.params(pass_net)
    grad = gradient(() -> loss(batch_Xp, batch_Xd, outcome), ps)
end

batches = Flux.Data.DataLoader((Xp_train,Xd_train,Y_train); batchsize=200, shuffle=false)
using BSON

function train!(cnn, data; nepochs=10)
    ps = Flux.params(cnn)
    opt = ADAM()
    for epoch in 1:nepochs
        @info "--------- $(epoch) ---------"
        for batch in data
            Xp = batch[1]
            Xd = batch[2]
            Y = batch[3]
            grad = gradient(() -> loss(Xp, Xd, Y), ps)
            Flux.Optimise.update!(opt, ps, grad)
            @info "Batch loss: $(loss(Xp, Xd, Y))"
        end
    end
    ps = Flux.params(cnn)
    bson("params.bson", ps=ps)
end

# train!(pass_net, batches; nepochs=3)
