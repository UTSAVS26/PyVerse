from data_utils import load_left_handed_data, compute_birth_year
from plot_utils import plot_gender_vs_age, plot_birthyear_vs_lefthanded

def main():
    df = load_left_handed_data()
    df = compute_birth_year(df)
    plot_gender_vs_age(df)
    plot_birthyear_vs_lefthanded(df)

if __name__ == "__main__":
    main()
