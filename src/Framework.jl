# Framework.jl
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
    
    # 闭包 Loss：固定 model 和 st
    core_loss(p, x, y) = loss_fn(p, x, y, model, st)

    dps = zero(ps_c)
    println("🚀 开始训练 [$label] ...")

    for epoch in 1:nepoch
        dps .= 0
        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Reverse),
            core_loss, Active,
            Duplicated(ps_c, dps), Const(X), Const(Y)
        )
        
        clamp!(dps, -0.5f0, 0.5f0) 
        opt_state, ps_c = Optimisers.update(opt_state, ps_c, dps)

        if epoch % step == 0 || epoch == nepoch
            yp, _, _ = predict_fn(model, ps_c, st, X, Y[:, 1])
            nse = of_NSE(Y, yp)
            l = core_loss(ps_c, X, Y)
            @printf "   [%s] Epoch %4d | Loss: %.6f | NSE: %.5f\n" label epoch l nse
        end
    end

    ypred, aux_out, _ = predict_fn(model, ps_c, st, X, Y[:, 1])
    final_nse = of_NSE(Y, ypred)
    
    return ypred, aux_out, ps_c, final_nse
end

export of_NSE
export train