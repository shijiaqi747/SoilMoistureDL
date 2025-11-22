using Lux, Random, Optimisers, ComponentArrays, Statistics, Printf
using MLUtils, Enzyme, RTableTools, Plots, NNlib
using SoilMoistureDL



begin
    f = "data/SM_AR_Batesville_8_WNW_2024.csv"
    df = fread(f)
    θ_all = Matrix(df[:, 3:end])' |> collect .|> Float32
    forcing_all = Matrix(df[:, [:P_CALC]])' |> collect .|> Float32

    # 时序切分 (70% 训练, 30% 测试)
    n_total = size(forcing_all, 2)
    split_idx = floor(Int, n_total * 0.70)

    X_train = forcing_all[:, 1:split_idx]
    Y_train = θ_all[:, 1:split_idx]

    X_test = forcing_all[:, split_idx:end]
    Y_test = θ_all[:, split_idx:end]
end


n_layer = size(Y_train, 1)
rng = Random.Xoshiro(42)

lateral_model = model_Lateral(; n_layers=n_layer, scale=0.01f0)
ps, st = Lux.setup(rng, lateral_model)# 初始化参数

# 传入不同的model，predict，loss_function
y_train, q_train, ps_trained, train_nse = train(X_train, Y_train, lateral_model, ps, st;
    predict_fn = predict_lateral, loss_fn = loss_lateral,         
    nepoch = 3000, lr = 0.002, label = "LatModel"            
)
println("\n>>> 训练集最终 NSE: $train_nse")


y_test, q_test, _ = predict_lateral(
    lateral_model, ps_trained, st, X_test, Y_test[:, 1]
)


test_nse = of_NSE(Y_test, y_test)
println(">>> 测试集 NSE: $test_nse")


 plot_lateral_results(
        θ_all,          # 整体观测数据 (Obs)
        y_train,        # 训练集预测 (Pred Train)
        y_test,         # 测试集预测 (Pred Test)
        q_train,        # 训练集通量
        q_test,         # 测试集通量
        split_idx;      # 切分点
        depth_labels = ["5cm", "10cm", "20cm", "50cm", "100cm"],
        save_path = "Result_Lateral_AllLayers.png"
    )
