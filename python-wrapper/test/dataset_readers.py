import dll
import unittest


class TestReaders(unittest.TestCase):
    """
    Test class for dataset readers
    """
    def test_mnist(self):
        """
        Method to test the mnist reader
        """
        reader = dll.PyMnistReader()
        reader.display()
        reader.display_pretty()
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
        """
        Method to test the text reader
        """
        reader = dll.PyTextReader("../dll/test/text_db/images", "../dll/test/text_db/labels")
        self.assertEqual(len(reader.read_images()), 9)
        self.assertEqual(len(reader.read_images()[0]), 784)
        self.assertEqual(reader.read_images()[5][155:160], [0, 77, 254, 107, 3])

        self.assertEqual(len(reader.read_labels()), 9)
        self.assertEqual(reader.read_labels()[0], 7)
