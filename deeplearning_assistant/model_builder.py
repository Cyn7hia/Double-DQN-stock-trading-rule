from os import path


# This is abstract class. You need to implement yours.
class AbstractModelBuilder:
    """
    generate
    """

    def __init__(self, weights_path=None):
        """
            :param weights_path:
        """
        self.weights_path = weights_path
        self.model = None

    @property
    def getModel(self):
        weights_path = self.weights_path
        if self.model == None:
            self.model = self.buildModel()

        if weights_path and path.isfile(weights_path):
            try:
                self.model.load_weights(weights_path)
            except Exception as e:
                print(e)

        return self.model

    # You need to override this method.
    def buildModel(self):
        raise NotImplementedError("You need to implement your own model.")
