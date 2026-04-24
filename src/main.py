import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

data = pd.read_csv("../data/covid_19_clean_complete.csv")
data.columns = data.columns.str.strip()
data['Date'] = pd.to_datetime(data['Date'])

def getDeathsPerMonth(month):
    filtered = data[data['Date'].dt.month == month]
    deaths_by_country = filtered.groupby('Country/Region')['Deaths'].sum()
    top3 = deaths_by_country.sort_values(ascending=False).head(3)
    return list(top3.items())

def getHighestRatio():
    overall = data.groupby('Country/Region')[['Confirmed', 'Deaths']].sum()
    overall['Ratio'] = overall['Deaths'] / overall['Confirmed'].replace(0, 1)
    max_country = overall['Ratio'].idxmax()
    max_ratio = overall['Ratio'].max()
    return max_country, max_ratio

def getHighestDay():
    confirmed_by_date = data.groupby('Date')['Confirmed'].max()
    max_date = confirmed_by_date.idxmax()
    max_cases = confirmed_by_date.max()
    return max_date, max_cases

def getTop10RatioChart():
    overall = data.groupby('Countr7y/Region')[['Confirmed', 'Deaths']].sum()
    overall['Ratio'] = overall['Deaths'] / overall['Confirmed'].replace(0, 1)
    top10 = overall.sort_values('Ratio', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(top10.index, top10['Ratio'], color='steelblue')
    plt.xlabel('Deaths to Confirmed Cases Ratio')
    plt.ylabel('Country')
    plt.title('Top 10 Countries by Deaths to Confirmed Cases Ratio')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def getTop5Confirmed():
    overall = data.groupby('Country/Region')['Confirmed'].max()
    top5 = overall.sort_values(ascending=False).head(5)
    return list(top5.items())

def plot_active_cases(country):
    plt.style.use('_mpl-gallery')
    country_df = data[data['Country/Region'] == country]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(country_df['Date'], country_df['Active'])

    ax.set(
        title=f"Active COVID-19 Cases Over Time - {country}",
        xlabel="Date",
        ylabel="Active Cases"
    )

    ax.grid(True, linestyle='--', alpha=0.5)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

def plot_heatmap():
    plt.style.use('_mpl-gallery')
    latest = data[data['Date'] == data['Date'].max()]

    if 'WHO Region' not in data.columns:
        print("WHO Region column not found in data.")
        return

    region_deaths = latest.groupby('WHO Region')['Deaths'].sum().sort_values()
    data_arr = region_deaths.values.reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data_arr, aspect='auto')

    ax.set(
        yticks=np.arange(len(region_deaths)),
        yticklabels=region_deaths.index,
        xticks=[],
        title="Global Death Density by WHO Region"
    )

    for i, val in enumerate(region_deaths.values):
        ax.text(0, i, f"{val:,}", ha='center', va='center')

    fig.colorbar(im)
    plt.tight_layout()
    plt.show()

def plot_fastest_growth():
    plt.style.use('_mpl-gallery')
    grouped = data.groupby('Country/Region')

    first = grouped.first()
    last = grouped.last()

    days = (data['Date'].max() - data['Date'].min()).days

    growth = pd.DataFrame({
        'first': first['Confirmed'],
        'last': last['Confirmed']
    })

    growth = growth[growth['first'] > 0]
    growth['daily_growth'] = (growth['last'] - growth['first']) / days

    top5 = growth.sort_values(by='daily_growth', ascending=False).head(5)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top5.index, top5['daily_growth'])

    ax.set(
        title="Top 5 Countries with Fastest Growth (Average Daily Cases)",
        xlabel="Average Daily Increase in Cases",
        ylabel="Country"
    )

    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def first_case_dates():
    plt.style.use('_mpl-gallery')

    if 'WHO Region' not in data.columns:
        print("WHO Region column not found in data.")
        return

    first_cases = data[data['Confirmed'] > 0] \
        .groupby(['Country/Region', 'WHO Region'])['Date'] \
        .min() \
        .reset_index()

    first_cases = first_cases.sort_values(by='Date')

    x = first_cases['Date']
    y = np.arange(len(first_cases))
    regions = first_cases['WHO Region'].astype('category').cat.codes

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, c=regions, s=40, alpha=0.7)

    ax.set(
        title="First Recorded Case of COVID-19 by Country",
        xlabel="Date of First Confirmed Case",
        ylabel="Countries"
    )

    fig.autofmt_xdate()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_recovery_rate():
    plt.style.use('_mpl-gallery')
    latest = data[data['Date'] == data['Date'].max()].copy()

    latest = latest[latest['Confirmed'] > 0]
    latest['Recovery Rate'] = latest['Recovered'] / latest['Confirmed']

    top5 = latest.sort_values(by='Recovery Rate', ascending=False).head(5)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top5['Country/Region'], top5['Recovery Rate'])

    ax.set(
        title="Top 5 Countries by Recovery Rate",
        xlabel="Recovery Rate",
        ylabel="Country"
    )

    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

while True:
    print("\nMenu:")
    print("1. Find country with most deaths in a specific month")
    print("2. Find country with highest deaths to confirmed cases ratio overall")
    print("3. Find the day with the most confirmed cases")
    print("4. View top 10 countries by deaths to confirmed ratio (bar chart)")
    print("5. Show top 5 countries with highest confirmed cases")
    print("6. View active cases over time for a country")
    print("7. View WHO region deaths heatmap")
    print("8. View top 5 countries with fastest growth")
    print("9. View first recorded case by country")
    print("10. View top 5 countries by recovery rate")
    print("11. Correlation between deaths and cases")
    print("12. Deaths to confirmed cases)")
    print("13. Lookup 5 countries")
    print("11. Exit")

    choice = input("Enter your choice (1-11): ").strip()

    if choice == '1':
        month_input = input("Enter a month (1-12 or month name e.g. January): ").strip()
        try:
            import calendar
            if month_input.isdigit():
                month = int(month_input)
            else:
                month = list(calendar.month_name).index(month_input.capitalize())
            top_countries = getDeathsPerMonth(month)
            print(f"Top 3 countries with most deaths in month {month}:")
            for rank, (country, deaths) in enumerate(top_countries, start=1):
                print(f"  {rank}. {country}: {deaths} deaths")
        except Exception as e:
            print("Error:", e)

    elif choice == '2':
        max_country, max_ratio = getHighestRatio()
        print(f"Country with highest deaths to confirmed cases ratio overall: {max_country} (Ratio: {max_ratio:.4f})")

    elif choice == '3':
        max_date, max_cases = getHighestDay()
        print(f"Day with the most confirmed cases: {max_date.strftime('%Y-%m-%d')} ({max_cases} cases)")

    elif choice == '4':
        getTop10RatioChart()

    elif choice == '5':
        top5 = getTop5Confirmed()
        print("Top 5 countries with highest confirmed cases:")
        for rank, (country, cases) in enumerate(top5, start=1):
            print(f"  {rank}. {country}: {cases:,} cases")

    elif choice == '6':
        country = input("Enter country name: ").strip()
        if country in data['Country/Region'].values:
            plot_active_cases(country)
        else:
            print("Country not found in data.")

    elif choice == '7':
        plot_heatmap()

    elif choice == '8':
        plot_fastest_growth()

    elif choice == '9':
        first_case_dates()

    elif choice == '10':
        plot_recovery_rate()

    elif choice == '11':
        Correleaton()

    elif choice == '12':
        DeathToConfirmed()

    elif choice == '13':
        Lookup5Countries()

    elif choice == '14':
        break

    else:
        print("Invalid choice. Please enter 1-11.")
