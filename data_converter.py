import os
import argparse
import json

class DataConverter:

    def __init__(self, dataset_dir, output_dir):

        path_to_json_files = os.path.join(dataset_dir, "data", "json")
        self.input_train_file = os.path.join(path_to_json_files, "train.json")
        self.input_test_file = os.path.join(path_to_json_files, "test.json")
        self.input_dev_file = os.path.join(path_to_json_files, "dev.json")

        self.output_dir = output_dir

        self.output_train_file = os.path.join(output_dir, "train.jsonl")
        self.output_test_file = os.path.join(output_dir, "test.jsonl")
        self.output_dev_file = os.path.join(output_dir, "dev.jsonl")

        self.glove_mapping = {
            '-LRB-': '(',
            '-RRB-': ')',
            '-LSB-': '[',
            '-RSB-': ']',
            '-LCB-': '{',
            '-RCB-': '}'
        }

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)

        self._convert_tacred_format_file(self.input_train_file, self.output_train_file)
        self._convert_tacred_format_file(self.input_dev_file, self.output_dev_file)
        self._convert_tacred_format_file(self.input_test_file, self.output_test_file)

    def _convert_tacred_format_file(self, input_file, output_file):
        with open(output_file, 'w') as output_file:
            for example in self._read_tacred_file(input_file):
                output_file.write(json.dumps(example) + "\n")
    
    def _read_tacred_file(self, input_file):
        with open(input_file, 'r') as input_file:
            input_examples = json.loads(input_file.readline())
            for input_example in input_examples:
                tokens = input_example['token']
                subj_offsets = (input_example['subj_start'], input_example['subj_end'] + 1)
                obj_offsets = (input_example['obj_start'], input_example['obj_end'] + 1)

                tokens = self.normalize_glove_tokens(tokens)

                output_example = {
                    "id": input_example['id'],
                    "tokens": tokens,
                    "label": input_example['relation'],
                    "entities": (subj_offsets, obj_offsets),
                    "grammar": ('SUBJ', 'OBJ'),
                    "type": (input_example['subj_type'], input_example['obj_type'])
                }

                yield output_example

    def normalize_glove_tokens(self, tokens):
        return [self.glove_mapping[token]
                if token in self.glove_mapping
                else token
                for token in tokens]


def main(args):
    assert os.path.exists(args.dataset_dir), "Input directory doesn't exist!"
    converter = DataConverter(args.dataset_dir, args.output_dir)
    converter.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=str, help="The root directory of the dataset")
    parser.add_argument('output_dir', type=str, help="An output directory of jsonl files")

    args = parser.parse_args()
    print(args)
    main(args)