
import pandas as pd

types = {'Flying': 0, 'Grass': 1, 'Fire': 2, 'Water': 3, 'Bug': 4, 'Normal': 5, 'Poison': 6, 'Electric': 7, 'Ground': 8, 'Fairy': 9, 'Fighting': 10, 'Psychic': 11, 'Rock': 12, 'Ghost': 13, 'Ice': 14, 'Dragon': 15, 'Dark': 16, 'Steel': 17}


def get_data():
    df = pd.read_csv('Pokemon.csv', header=0)

    dft = df.T

    # df = df[df.Legendary == False]

    inputs = []

    for i in range(len(df)):
        pokemon = []
        pokemon.append(dft[i]['HP'])
        pokemon.append(dft[i]['Attack'])
        pokemon.append(dft[i]['Defense'])
        pokemon.append(dft[i]['Sp. Atk'])
        pokemon.append(dft[i]['Sp. Def'])
        pokemon.append(dft[i]['Speed'])
        inputs.append(pokemon)

    outputs = [[0]*18 for i in range(len(df))]

    for i in range(len(df)):
        column = types[dft[i]['Type 1']]
        outputs[i][column] = 1

    return inputs, outputs
