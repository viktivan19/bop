import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyse_results():
    """
    Read the survey results as CSV and produce analysis by calculating statistics and visualising the data.
    :return: None
    """
    df = pd.read_excel('../SurveyReport-10615648-12-06-2022-T141042.019.xlsx', sheet_name='Raw Data')
    df = df[df['Response Status']=='Completed']
    tim_to_comp = df['Time Taken to Complete (Seconds)'].mean() / 60

    df = df.drop(['Seq. Number', 'External Reference',
       'Custom Variable 1', 'Custom Variable 2', 'Custom Variable 3',
       'Custom Variable 4', 'Custom Variable 5', 'Respondent Email',
       'Email List',], axis=1)

    answers = df[[       'What do you think of this melody?   Clip 1',
       'What do you think of this melody?   Clip 2',
       'What do you think of this melody?   Clip 3',
       'What do you think of this melody?   Clip 4',
       'What do you think of this melody?   Clip 5',
       'What do you think of this melody?   Clip 6',
       'What do you think of this melody?   Clip 7',
       'Did you remember this melody from the first round? ',
       'Did you remember this melody from the first round? .1',
       'Did you remember this melody from the first round? .2',
       'Did you remember this melody from the first round? .3',
       'Did you remember this melody from the first round? .4',
       'Did you remember this melody from the first round? .5',
       'Did you remember this melody from the first round? .6', 'Response ID']]

    answers = answers.rename({'Did you remember this melody from the first round? ': 'Did you remember this melody from the first round? - Clip 1',
       'Did you remember this melody from the first round? .1':'Did you remember this melody from the first round? - Clip 2',
       'Did you remember this melody from the first round? .2':'Did you remember this melody from the first round? - Clip 3',
       'Did you remember this melody from the first round? .3':'Did you remember this melody from the first round? - Clip 4',
       'Did you remember this melody from the first round? .4':'Did you remember this melody from the first round? - Clip 5',
       'Did you remember this melody from the first round? .5':'Did you remember this melody from the first round? - Clip 6',
       'Did you remember this melody from the first round? .6':'Did you remember this melody from the first round? - Clip 7',
                              'What do you think of this melody?   Clip 1':'What do you think of this melody?  - Clip 1',
                             'What do you think of this melody?   Clip 2':'What do you think of this melody?  - Clip 2',
                             'What do you think of this melody?   Clip 3':'What do you think of this melody?  - Clip 3',
                             'What do you think of this melody?   Clip 4':'What do you think of this melody?  - Clip 4',
                             'What do you think of this melody?   Clip 5':'What do you think of this melody?  - Clip 5',
                             'What do you think of this melody?   Clip 6':'What do you think of this melody?  - Clip 6',
                             'What do you think of this melody?   Clip 7':'What do you think of this melody?  - Clip 7',
                              }, axis=1)

    missing = answers.isna().sum()
    missing = pd.DataFrame(missing)
    missing = missing.rename({0:'Number of Missing Answers'}, axis=1)
    total_missing = missing.sum()

    n_missing = answers.notnull().sum()
    total_not_missing = n_missing.sum()



    first = answers[['What do you think of this melody?  - Clip 1',
       'What do you think of this melody?  - Clip 2',
       'What do you think of this melody?  - Clip 3',
       'What do you think of this melody?  - Clip 4',
       'What do you think of this melody?  - Clip 5',
       'What do you think of this melody?  - Clip 6',
       'What do you think of this melody?  - Clip 7',
                     'Response ID']]
    first.describe().transpose()
    first = first.melt(id_vars=['Response ID'])
    first[['question', 'clip number']] = first['variable'].str.split('-', expand=True)
    first = first.rename({ 'value': 'first_answer'}, axis=1)
    first = first.drop(['variable','question'], axis=1)

    # average_response_means = first.drop('Response ID', axis=1).transpose().describe()
    #
    # sns.histplot(data=average_response_means, x='average score', color='skyblue')
    # plt.title("Distribution of Average Score Across Clips by Participants")
    # plt.show()


    second = answers[['Did you remember this melody from the first round? - Clip 1',
       'Did you remember this melody from the first round? - Clip 2',
       'Did you remember this melody from the first round? - Clip 3',
       'Did you remember this melody from the first round? - Clip 4',
       'Did you remember this melody from the first round? - Clip 5',
       'Did you remember this melody from the first round? - Clip 6',
       'Did you remember this melody from the first round? - Clip 7', 'Response ID']]
    second.describe().transpose()
    second = second.melt(id_vars=[ 'Response ID'])
    second[['question', 'clip number']] = second['variable'].str.split('-', expand=True)

    second_count = second.replace({1.: "no recall", 2.: 'some recall', 3.: 'total recall'})
    second_count = second_count.drop('Response ID', axis=1)

    (second_count == 'no recall').sum().sum()


    second = second.rename({'value': 'second_answer'}, axis=1)
    second = second.drop(['variable', 'question'], axis=1)

    full = first.merge(second, on=['Response ID', 'clip number'], how='outer')

    full['second_answer'] =  full['second_answer'].astype(float)
    sns.scatterplot(data=full, x="first_answer", y="second_answer", hue="clip number")
    plt.show()

    sns.violinplot(data=full, x="clip number", y="first_answer", palette='crest')
    plt.title('Distribution of pleasantness scores per audio clip')
    plt.show()

    sns.barplot(data=full, x='clip number', y='second_answer')
    plt.show()

    full = full.replace({'second_answer': {1.: "no recall", 2.: 'some recall', 3.: 'total recall'}})

    sns.histplot(data=full, x='clip number', hue='second_answer', multiple='fill', palette='dark:salmon_r')
    plt.title('Memorability of each audio clip')
    plt.show()

    avg_pl = full.groupby('clip number')['first_answer'].mean().reset_index().rename({'first_answer': 'average pleasantness'},axis=1)
    full = full.merge(avg_pl, on='clip number', how='left')


    sns.scatterplot(data=full, x='second_answer', y='first_answer')
    plt.show()

    sns.boxplot(data=full, x='second_answer', y='first_answer', palette='crest')
    plt.title('Pleasantness score distribution vs memorability')
    plt.ylabel('Pleasantness Score')
    plt.xlabel('Memorability Score')
    plt.show()

    return None



analyse_results()