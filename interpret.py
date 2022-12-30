import torch


def feature_occlusion(model, input_vec: torch.Tensor, debug=False):
    '''
    用feature occlsion的方法对于模型的一个输入输出结果进行解释，
    生成一个热图来反映输入的每一部分对输出的影响程度。
    '''
    if debug:
        print(input_vec.shape)
    length = input_vec.shape[1]
    output = model(input_vec)
    heatmap = []
    for i in range(length):
        mask = torch.ones([1, length, 1]).cuda()
        mask[0][i][0] = 0
        new_input = input_vec*mask
        new_output = model(new_input)
        if debug:
            print(new_input)
            print(new_output)
        diff: torch.Tensor = output-new_output
        heatmap.append(diff.item())
    if debug:
        print(heatmap, len(heatmap))
    return heatmap

    