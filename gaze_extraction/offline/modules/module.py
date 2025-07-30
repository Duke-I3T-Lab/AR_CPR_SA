from offline.data import GazeData

class Module:
    def __init__(self) -> None:
        pass

    def update(self, data: GazeData) -> GazeData:
        raise NotImplementedError('Subclasses of Module should implement update.')