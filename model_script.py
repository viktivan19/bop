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
            next_note = None
            elements = instrument.partitionByInstrument(midi)
            if elements: # file has instrument elements
                next_note = elements.elements[0].recurse()
            else: # file has notes in a flat structure
                next_note = midi.flat.notes
            for element in next_note:
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
    pitch_list = sorted(set(item for item in notes))
    # create a dictionary to map pitches to integers
    numeric_notes = dict((note, number) for number, note in enumerate(pitch_list))
    network_input = []
    network_output = []
    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([numeric_notes[char] for char in sequence_in])
        network_output.append(numeric_notes[sequence_out])
    num_tunes = len(network_input)
    # reshape the input into a format compatible with LSTM layers
    network_input_train = np.reshape(network_input, (num_tunes, sequence_length, 1))
    # normalize input

    num_elements = len(set(notes))

    network_input_train = network_input_train / float(num_elements)
    network_output = np_utils.to_categorical(network_output)

    return network_input_train, network_output, num_elements, network_input, pitch_list, numeric_notes


def train_model(network_input_train, network_output, num_elements):
    """
    Train LSTM model
    :param network_input_train: input sequences
    :param network_output: output sequences
    :param num_elements: number of pitches in training
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
    model.add(Dense(num_elements))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    print( "Start time: ", datetime.now())

    model.fit(network_input_train, network_output, epochs=2, batch_size=128)

    print("Finish time: ", datetime.now())

    return model


def get_predictions(model, network_input, pitch_list, num_elements, numeric_notes):
    """
    generate melodies from model
    :param model: Fitted model
    :param network_input: input sequences
    :param pitch_list: list of pitches from the training data
    :param num_elements:  number of pitches in training
    :return:
    """

    for song in range(10):

        start = np.random.randint(0, len(network_input) - 1)
        preds = []
        pattern = network_input[start]
        for note_index in range(40):
            int_to_note = dict((number, note) for number, note in enumerate(pitch_list))
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(num_elements)
            prediction = model.predict(prediction_input, verbose=0)

            index = np.argsort(np.max(prediction, axis=0))[-2]
            single_pred = int_to_note[index]
            preds.append(single_pred)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]

            pred = np.argsort(prediction[0])
            for notes in pred:
                single_pred = int_to_note[notes]
                preds.append(single_pred)

        offset = 0
        output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in preds:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            element_chord = pattern.split('.')
            notes = []
            for current_note in element_chord:
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
    network_input_train, network_output, num_elements, network_input, pitch_list, numeric_notes = generate_input_sequences(notes)
    model = train_model(network_input_train, network_output, num_elements)
    output_notes, song = get_predictions(model, network_input, pitch_list, num_elements, numeric_notes)

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output_file.mid'.format(song))


