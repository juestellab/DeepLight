import tomli

def load_paths(config_file):
    with open(config_file) as f:
        return tomli.load(f)['paths']
