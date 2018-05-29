from gan.net import ConvGenerator, ConvDiscriminator, MnistClassifier
from gan.gan import GAN
from gan.cgan import CGAN
from gan.gan_classifier import GANClassifier
from gan.function import MnistGANloader, MnistGANClassloader


gen = ConvGenerator([7*7*128], [7, 7, 128], [(64, 4, 2), (1, 4, 2)])
dis = ConvDiscriminator([(64, 4, 2), (128, 4, 2)])
# clf = MnistClassifier()

# model = GAN(gen, dis, [28, 28, 1], 100)
# model.train("Datas/mnist", data_loader=MnistGANloader)

# model = GANClassifier(gen, dis, clf, [28, 28, 1], 10, 100)
model = CGAN(gen, dis, [28, 28, 1], 10, 100)
model.train("Datas/mnist", data_loader=MnistGANClassloader)

# # image folder
# from gan.net import ConvGenerator, ConvDiscriminator
# from gan.gan import GAN
# from gan.wgan import WGAN
# gen = ConvGenerator([12*12*192], [12, 12, 192], [(96, 4, 2), (24, 4, 2), (3, 4, 2)], keep_dropout_p=1)
# dis = ConvDiscriminator([(24, 4, 2), (96, 4, 2), (192, 4, 2)], keep_dropout_p=1)
#
# # model = GAN(gen, dis, [96, 96, 3], 100)
# model = WGAN(gen, dis, [96, 96, 3], 100)
# model.train("Datas/faces")
