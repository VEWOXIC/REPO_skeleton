class xxxmodel(object):
    def __init__(self, model_setting) -> None:
        pass
    def forward(self,x):
        x = x.reshape(BLC)
        return x