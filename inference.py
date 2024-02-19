from tools.ChartVLM import infer_ChartVLM

if __name__ == '__main__':
    model = '${PATH_TO_PRETRAINED_MODEL}/ChartVLM/base/'  #${PATH_TO_PRETRAINED_MODEL}
    image = './assets/test.png'  
    text = 'who has the largest value?'

    output = infer_ChartVLM(image, text, model)

    print(output)
