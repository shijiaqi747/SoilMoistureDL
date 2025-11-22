# Model_Lateral.jl
using Lux, NNlib, Statistics, Random

struct PhysicalLateralLayer{M} <: Lux.AbstractLuxLayer
    k_model::M
    scale::Float32
end

Lux.initialparameters(rng::AbstractRNG, m::PhysicalLateralLayer) = Lux.initialparameters(rng, m.k_model)
Lux.initialstates(rng::AbstractRNG, m::PhysicalLateralLayer) = Lux.initialstates(rng, m.k_model)

function (m::PhysicalLateralLayer)(x, ps, st)
    k, st_new = m.k_model(x, ps, st)
    q_lat = - k .* x .* m.scale 
    return q_lat, st_new
end


"""
build_vertical_net
改进：scale 参数改为向量，允许深层变化得比浅层慢
"""
function build_vertical_net(; n_layers, n_in, scale=nothing)
    in_dim = n_layers + n_in
    
    # 如果没有指定具体的 scale 向量，则使用默认策略
    # 表层快 (0.02)，深层慢 (0.002)
    if isnothing(scale)
        # 假设 5 层: [5cm, 10cm, 20cm, 50cm, 100cm]
        # 深层变化极慢，必须把 scale 压下去，否则会震荡
        scale_vec = Float32[0.02, 0.02, 0.01, 0.005, 0.002] 
    else
        scale_vec = Float32.(scale)
    end

    Chain(
        Dense(in_dim => 64, relu), 
        Dense(64 => 64, relu),     
        Dense(64 => n_layers, init_weight=Lux.zeros32, init_bias=Lux.zeros32), 
        # 关键：使用 Element-wise 乘法，每一层乘不同的系数
        x -> x .* scale_vec 
    )
end

function build_lateral_net(; n_layers, scale=nothing) 
    # 侧向流同样：表层侧向流强，深层侧向流弱
    if isnothing(scale)
        # 深层 (Layer 5) 几乎没有侧向流，给极小的 scale
        scale_vec = Float32[0.01, 0.01, 0.005, 0.001, 0.0001]
    else
        scale_vec = Float32.(scale)
    end

    k_net = Chain(
        Dense(n_layers => 32, tanh), 
        Dense(32 => n_layers), 
        x -> softplus.(x) 
    )
    # PhysicalLateralLayer 需要修改一下以支持向量 scale
    return PhysicalLateralLayer(k_net, scale_vec)
end


"""
    model_Lateral(; n_layers, scale=0.01f0)
直接返回构建好的侧向流耦合模型结构。
"""
function model_Lateral(; n_layers, scale=0.01f0)
    # 这里定义了 vert 和 lat 是如何组合的
    return (
        vert = build_vertical_net(; n_layers=n_layers, scale=scale), 
        lat  = build_lateral_net(; n_layers=n_layers, scale=scale)  
    )
end


function predict_lateral(model, ps, st, forcing, θ_init)
    T = eltype(θ_init)
    n_time = size(forcing, 2)
    n_layer = length(θ_init)
    h_preds = zeros(T, n_layer, n_time)
    q_lat_preds = zeros(T, n_layer, n_time)
    
    h_preds[:, 1] .= θ_init
    st_curr = st

    for t in 1:n_time-1
        h_prev = h_preds[:, t]
        u_curr = forcing[:, t+1]

        x_in = vcat(h_prev, u_curr)
        dh_vert, st_vert = model.vert(x_in, ps.vert, st_curr.vert)
        dh_lat, st_lat = model.lat(h_prev, ps.lat, st_curr.lat)

        r = h_prev .+ dh_vert .+ dh_lat
        r = clamp.(r, 0.0f0, 0.8f0) 
        
        h_preds[:, t+1] .= r
        q_lat_preds[:, t+1] .= dh_lat
        st_curr = (vert=st_vert, lat=st_lat)
    end
    return h_preds, q_lat_preds, st_curr
end

function loss_lateral(p, x, y, model, st; reg_weight=0.05f0)
    h_pred, q_lat, _ = predict_lateral(model, p, st, x, y[:, 1])
    loss_mse = mean((h_pred .- y) .^ 2)
    loss_reg = mean(-q_lat) * reg_weight 
    return loss_mse + loss_reg
end

function loss_lateral_weighted(p, x, y, model, st; reg_weight=0.05f0)
    h_pred, q_lat, _ = predict_lateral(model, p, st, x, y[:, 1])
    
    # 定义层权重：越深，权重越大 (因为深层数值波动小，容易被忽略)
    # Layer 1-5 权重: [1, 1, 2, 5, 10]
    layer_weights = Float32[1.0, 1.0, 2.0, 5.0, 10.0]
    
    # 计算加权 MSE
    # (h_pred .- y).^2 得到 [5, Time] 矩阵
    # layer_weights 是 [5] 向量，通过广播机制相乘
    weighted_sq_diff = (h_pred .- y).^2 .* layer_weights
    loss_mse = mean(weighted_sq_diff)
    
    loss_reg = mean(-q_lat) * reg_weight 
    return loss_mse + loss_reg
end


export PhysicalLateralLayer
export build_vertical_net, build_lateral_net, model_Lateral

export predict_lateral
export loss_lateral, loss_lateral_weighted