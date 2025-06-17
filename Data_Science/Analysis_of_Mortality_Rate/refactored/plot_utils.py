import matplotlib.pyplot as plt

def plot_gender_vs_age(df):
    fig, ax = plt.subplots()
    ax.plot('Age', 'Female', data=df, marker='o')
    ax.plot('Age', 'Male', data=df, marker='x')
    ax.set_xlabel('Age')
    ax.set_ylabel('Percentage Left-Handed')
    ax.legend()
    plt.show()

def plot_birthyear_vs_lefthanded(df):
    df['Mean_lh'] = df[['Female', 'Male']].mean(axis=1)
    fig, ax = plt.subplots()
    ax.plot('Birth_year', 'Mean_lh', data=df)
    ax.set_xlabel('Year of birth')
    ax.set_ylabel('Percentage left-handed')
    plt.show()
