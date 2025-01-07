unknown_X = data.load_predicting_data(args)
sand_thickness_preds = model(unknown_X.coords, unknown_X.features)
unknown_X.coords = unknown_X.coords.to(args.device)

result = torch.cat((unknown_X.coords, sand_thickness_preds), dim=1)
# 将结果转换为 NumPy 数组
result_np = result.detach().to("cpu").numpy()
# 将 NumPy 数组保存到文本文件
np.savetxt(args.Spatial_data_predicted, result_np, fmt='%f')  # 'result.txt' 是保存的文件名，'%f' 表示保存浮点数
