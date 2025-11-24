# src/Framework.jl
using Lux, Optimisers, ComponentArrays, Enzyme, Statistics, Printf

function of_NSE(obs, sim)
    top = sum((sim .- obs) .^ 2)
    bot = sum((obs .- mean(obs)) .^ 2)
    return 1.0f0 - (top / bot)
end

"""
    train(X, Y, model, ps, st; predict_fn, loss_fn, ...)
通用训练函数。
"""
function train(X, Y, model, ps, st; 
                        predict_fn::Function, 
                        loss_fn::Function,
                        lr=0.002, nepoch=3000, step=100, label="Model")
    
    ps_c = ComponentArray(ps)
    opt = Optimisers.ADAM(lr) 
    opt_state = Optimisers.setup(opt, ps_c)
    
    # 1. 定义 core_loss 接受 5 个参数
    function core_loss(p, x, y, m, s)
        return loss_fn(p, x, y, m, s)
    end

    dps = zero(ps_c)
    println("🚀 开始训练 [$label] ...")

    for epoch in 1:nepoch
        dps .= 0
        
        # 2. 关键修复：autodiff 必须传入对应的 5 个参数
        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Reverse),
            core_loss, Active,
            Duplicated(ps_c, dps), # 参数 1: p (求导)
            Const(X),              # 参数 2: x (常量)
            Const(Y),              # 参数 3: y (常量)
            Const(model),          # 参数 4: m (常量) <--- 补上这里
            Const(st)              # 参数 5: s (常量) <--- 补上这里
        )
        
        clamp!(dps, -0.5f0, 0.5f0) 
        opt_state, ps_c = Optimisers.update(opt_state, ps_c, dps)

        if epoch % step == 0 || epoch == nepoch
            # 计算 Loss 用于显示 (这里您之前写对了，保持 5 个参数)
            l = core_loss(ps_c, X, Y, model, st) 

            yp, _, _ = predict_fn(model, ps_c, st, X, Y[:, 1])
            nse = of_NSE(Y, yp)
            
            @printf "   [%s] Epoch %4d | Loss: %.6f | NSE: %.5f\n" label epoch l nse
        end
    end

    ypred, aux_out, _ = predict_fn(model, ps_c, st, X, Y[:, 1])
    final_nse = of_NSE(Y, ypred)
    
    return ypred, aux_out, ps_c, final_nse
end

export of_NSE
export train