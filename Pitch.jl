module Pitch

using Plots, Images

export draw_pitch!, draw_probs


##### Module for plotting a football pitch

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

function draw_probs(sample_input, model)
    pred_surface = imfilter(model(sample_input)[:,:,1,1], Kernel.gaussian(0.5))'
    xs=LinRange(0,120,52)
    ys=LinRange(0,80,34)
    p = plot(aspect_ratio=:equal,axis=nothing,border=:none)
    contour!(xs, ys, pred_surface, fill=true, seriescolor=:summer, linecolor=:black, linewidth=0.5, colorbar=false)
    x = findmax(sample_input[:,:,1])[2][1] - 1
    y = findmax(sample_input[:,:,1])[2][2] - 1
    scatter!([x*120/52],[y*80/34], color=:red, markersize=5, label=nothing)
    draw_pitch!()
    return p, pred_surface
end

end