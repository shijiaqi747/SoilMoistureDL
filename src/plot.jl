"""
    plot_lateral_results(Y_obs, y_train, y_test, q_train, q_test, split_idx; 
                         depth_labels, save_path)

绘制侧向流模型的多层验证结果图（含训练集/测试集分割）。
- 左列：土壤含水量拟合 (Obs vs Pred)
- 右列：推断出的侧向通量 (Lateral Flux)
"""
function plot_lateral_results(Y_obs, y_train, y_test, q_train, q_test, split_idx;
                              depth_labels=["5cm", "10cm", "20cm", "50cm", "100cm"],
                              save_path="Result_Lateral_AllLayers.png")
    
    println("\n>>> 正在生成绘图 $save_path ...")

    # 获取维度信息
    n_layer = size(Y_obs, 1)
    n_total = size(Y_obs, 2)
    
    # 定义时间轴
    t_train = 1:split_idx
    t_test = split_idx:n_total
    
    plots_list = []
    
    for i in 1:n_layer
        # 1. 动态计算 NSE (确保维度匹配)
        # 训练集: 对应 Y_obs 的前 split_idx 个数据
        obs_tr = Y_obs[i, 1:size(y_train, 2)]
        nse_tr = of_NSE(obs_tr, y_train[i, :])
        
        # 测试集: 对应 Y_obs 从 split_idx 开始的数据
        obs_te = Y_obs[i, split_idx:end]
        nse_te = of_NSE(obs_te, y_test[i, :])
        
        # 2. 左图: SM 拟合
        # 仅在第一行显示图例标签，防止混乱
        lbl_obs   = (i == 1) ? "Obs" : ""
        lbl_train = (i == 1) ? "Train" : ""
        lbl_test  = (i == 1) ? "Test" : ""
        
        # 仅在第一行显示标题
        title_sm = (i == 1) ? "Soil Moisture Fitting" : ""
        title_flux = (i == 1) ? "Inferred Lateral Flux" : ""

        # 为了避免 Plotly/GR 后端对 length 不一致的警告，显式截取 t_train/t_test 的长度
        # (通常 y_train 的长度就是 split_idx，但为了稳健性)
        t_tr_range = t_train[1:size(y_train, 2)]
        t_te_range = t_test[1:size(y_test, 2)]

        p_sm = plot(title=title_sm, ylabel="$(depth_labels[i]) SM", margin=3Plots.mm)
        plot!(p_sm, 1:n_total, Y_obs[i, :], label=lbl_obs, c=:black, lw=1.5, alpha=0.3)
        plot!(p_sm, t_tr_range, y_train[i, :], label=lbl_train, c=:red, lw=1.5)
        plot!(p_sm, t_te_range, y_test[i, :], label=lbl_test, c=:green, lw=1.5)
        
        # 分割线
        vline!(p_sm, [split_idx], c=:gray, ls=:dash, label="")
        
        # 标注 NSE
        annotate!(p_sm, split_idx, maximum(Y_obs[i, :]), 
                  text("Tr=$(round(nse_tr,digits=2))\nTe=$(round(nse_te,digits=2))", 8, :center, :top))
        
        # 3. 右图: 侧向流 Flux
        lbl_tr_flux = (i == 1) ? "Train Flux" : ""
        lbl_te_flux = (i == 1) ? "Test Flux" : ""
        
        p_flux = plot(title=title_flux, ylabel="Flux")
        plot!(p_flux, t_tr_range, q_train[i, :], label=lbl_tr_flux, c=:red, alpha=0.6, lw=1)
        plot!(p_flux, t_te_range, q_test[i, :], label=lbl_te_flux, c=:green, alpha=0.6, lw=1)
        vline!(p_flux, [split_idx], c=:gray, ls=:dash, label="")
        
        push!(plots_list, p_sm)
        push!(plots_list, p_flux)
    end
    
    # 4. 组合大图 (n_layer 行 x 2 列)
    # 动态调整高度: 每层给 250px 高度
    plot_height = n_layer * 250
    final_layout = plot(plots_list..., layout=(n_layer, 2), size=(1000, plot_height))
    
    display(final_layout)
    savefig(save_path)
    println("绘图完成！已保存至: $save_path")
end

