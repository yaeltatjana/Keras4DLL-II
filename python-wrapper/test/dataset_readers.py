import dll
import unittest


def test_mnist():
    reader = dll.PyMnistReader()
    reader.display()
    reader.display_pretty()


def test_text():
    # nothing to be done
    print()


class TestReaders(unittest.TestCase):
    def test_mnist(self):
        reader = dll.PyMnistReader()
        self.assertEqual(len(reader.read_dataset()), 4)

        self.assertEqual(len(reader.read_dataset()['training_images']), 60000)
        self.assertEqual(len(reader.read_dataset()['training_images'][0]), 784)
        self.assertEqual(reader.read_dataset()['training_images'][5][155:160], [13, 25, 100, 122, 7])

        self.assertEqual(len(reader.read_dataset()['test_images']), 10000)
        self.assertEqual(len(reader.read_dataset()['test_images'][0]), 784)
        self.assertEqual(reader.read_dataset()['test_images'][5][155:160], [0, 77, 254, 107, 3])

        self.assertEqual(len(reader.read_dataset()['training_labels']), 60000)
        self.assertEqual(reader.read_dataset()['training_labels'][:5], [5, 0, 4, 1, 9])

        self.assertEqual(len(reader.read_dataset()['test_labels']), 10000)
        self.assertEqual(reader.read_dataset()['test_labels'][:5], [7, 2, 1, 0, 4])

    def test_text(self):
        reader = dll.PyTextReader("../dll/test/text_db/images", "../dll/test/text_db/labels")
        self.assertEqual(len(reader.read_images()), 9)
        self.assertEqual(len(reader.read_images()[0]), 784)
        self.assertEqual(reader.read_images()[5][155:160], [0, 77, 254, 107, 3])

        self.assertEqual(len(reader.read_labels()), 9)
        self.assertEqual(reader.read_labels()[0], 7)
