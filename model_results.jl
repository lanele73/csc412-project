using JLD2, BSON, Flux, Zygote, Plots, Images, Random
using Statistics: mean
include("Pitch.jl")
using .Pitch
include("cnn.jl")
using .cnn


##### Load model #####


function unpack(path)
    JLD2.@load path data
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


### Look at performance

let
    plot(title = "Training loss", ylabel="Loss", xlabel="Epoch")
    batch_range = 1/num_batches:(1/num_batches):size(epoch_loss_values)[1]
    epoch_range = 1:size(epoch_loss_values)[1]
    plot!(batch_range,batch_loss_values, label="Batch")
    plot!(epoch_range,epoch_loss_values, label="Epoch", linewidth=3)
end


begin
    sample_index = rand(MersenneTwister(1), (1:size(Y_test)[1]), 4)
    sample_input = Xp_test[:,:,:,sample_index]

    plots = []
    surfaces = []
    for i in 1:4
        p, surf = draw_probs(sample_input[:,:,:,i:i], conv_net)
        push!(surfaces, surf)
        push!(plots, p)
    end
    clim = (minimum(minimum.(surfaces)), maximum(maximum.(surfaces)))
    h2 = scatter([0,0], [0,1], zcolor=[0,1], xlims=(1,1.1), label="",clims=clim, c=:summer, framestyle=:none)
    l = @layout [grid(2, 2) a{0.035w}]
    plot(plots..., h2, layout=l)
end