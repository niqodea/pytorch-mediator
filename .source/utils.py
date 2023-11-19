def preprocess(input):
    return input.log()


def postprocess(input):
    return input.exp()
