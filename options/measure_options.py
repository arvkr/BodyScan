from .train_options import TrainOptions
class MeasureOptions(TrainOptions):
    def initialize(self):
        TrainOptions.initialize(self)
        self.parser.add_argument('--image_name', type=str, default='./data/inputs/11.jpg', help='Name of input image')