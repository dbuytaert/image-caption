import unittest
from caption import clean_caption

class TestCleanCaption(unittest.TestCase):
    def test_removes_image_prefix(self):
        test_cases = [
            (
                "This is an image of a cat sleeping on a windowsill.",
                "A cat sleeping on a windowsill."
            ),
            (
                "This image shows a ginger cat playing with yarn.",
                "A ginger cat playing with yarn."
            ),
            (
                "The picture contains two cats grooming each other.",
                "Two cats grooming each other."
            ),
            (
                "A black cat in a cardboard box.", # Already clean
                "A black cat in a cardboard box."
            ),
            (
                "this is a photograph of three kittens drinking milk",
                "Three kittens drinking milk."
            ),
            (
                "The scene depicts a cat chasing a laser pointer",
                "A cat chasing a laser pointer."
            ),
            # Adding quote-related test cases with cats
            (
                '"a sleepy cat curled up in a sunbeam"',
                "A sleepy cat curled up in a sunbeam."
            ),
            (
                '\"This is an image of a cat climbing a tree\"',
                "A cat climbing a tree."
            ),
            (
                "\"The picture shows two cats watching birds\"",
                "Two cats watching birds."
            )
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = clean_caption(input_text)
                self.assertEqual(result, expected, f"\nInput: {input_text}\nExpected: {expected}\nGot: {result}")

if __name__ == '__main__':
    unittest.main()