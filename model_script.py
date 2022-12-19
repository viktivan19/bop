from music21 import converter, instrument, note as m_note, chord, stream
from os import listdir
from os.path import isfile, join
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from datetime import datetime


def process_data():
    """
    Read files and process into note sequences.
    :return: list of note sequences.
    """

    notes = []

    for file in [f for f in listdir("../data_final") if isfile(join("../data_final", f))]:
        print('Processing {}'.format(file))
        if (file != '.DS_Store') & (file != '.gitkeep'):
            midi = converter.parse("../data_final/" + file)
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts: # file has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, m_note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes


def generate_input_sequences(notes):
    """
    Generate training input and output sequences
    :param notes: sequence of raw notes
    :return: sequences ready to model training
    """
    sequence_length = 46
    # get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    network_input = []
    network_output = []
    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
    n_patterns = len(network_input)
    # reshape the input into a format compatible with LSTM layers
    network_input_train = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input

    n_vocab = len(set(notes))

    network_input_train = network_input_train / float(n_vocab)
    network_output = np_utils.to_categorical(network_output)

    return network_input_train, network_output, n_vocab, network_input, pitchnames, note_to_int


def train_model(network_input_train, network_output, n_vocab):
    """
    Train LSTM model
    :param network_input_train: input sequences
    :param network_output: output sequences
    :param n_vocab: number of pitches in training
    :return: fitted model
    """

    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input_train.shape[1], network_input_train.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3, ))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    print( "Start time: ", datetime.now())

    model.fit(network_input_train, network_output, epochs=2, batch_size=128)

    print("Finish time: ", datetime.now())

    return model


def get_predictions(model, network_input, pitchnames, n_vocab, note_to_int):
    """
    generate melodies from model
    :param model: Fitted model
    :param network_input: input sequences
    :param pitchnames: list of pitches from the training data
    :param n_vocab:  number of pitches in training
    :return:
    """

    for song in range(10):

        start = np.random.randint(0, len(network_input) - 1)
        prediction_output = []
        pattern = network_input[start]
        for note_index in range(40):
            int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            prediction = model.predict(prediction_input, verbose=0)

            index = np.argsort(np.max(prediction, axis=0))[-2]
            result = int_to_note[index]
            prediction_output.append(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
            start = note_to_int[result]


            # test
            pred = np.argsort(prediction[0])
            for notes in pred:
                result = int_to_note[notes]
                prediction_output.append(result)


        offset = 0
        output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = m_note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = m_note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    return output_notes, song


if __name__ == "__main__":
    notes = process_data()
    network_input_train, network_output, n_vocab, network_input, pitchnames, note_to_int = generate_input_sequences(notes)
    model = train_model(network_input_train, network_output, n_vocab)
    output_notes, song = get_predictions(model, network_input, pitchnames, n_vocab, note_to_int)

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output_file.mid'.format(song))

