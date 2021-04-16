using BSON, Flux, Zygote, Plots, Images


##### Load model #####

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

load_model = true
if load_model
    using Zygote
    BSON.@load "saved_runs/params25.bson" ps
    BSON.@load "saved_runs/loss_history25.bson" history
    batch_loss_values, epoch_loss_values = history
    Flux.loadparams!(conv_net, ps)
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
    contour!(xs, ys, pred_surface, fill=true, seriescolor=:summer, linecolor=:black, linewidth=0.5)
    # heatmap!(xs, ys, pred_surface, seriescolor=:summer)
    x = findmax(sample_input[:,:,1])[2][1] - 1
    y = findmax(sample_input[:,:,1])[2][2] - 1
    scatter!([x*120/52],[y*80/34], color=:red, markersize=5, label=nothing)
    draw_pitch!()
    return p
end

### Look at samples
begin
    sample_index = rand(MersenneTwister(1), (1:size(Y_test)[1]), 4)
    sample_input = Xp_test[:,:,:,sample_index] 
    pred_surfaces = conv_net(sample_input)

    plots = []
    for i in 1:4
        p = draw_probs(sample_input[:,:,:,i:i])
        push!(plots, p)
    end
    plot(plots..., layout=(2,2))
end