import matplotlib.pyplot as plt
import seaborn as sns


def plot_csat_distibution(df):
    fig, ax = plt.subplots(figsize=(10,4))
    sns.histplot(df['CSAT Score'], bins=30, color='steelblue', kde=True)
    ax.set_title('Distribution of CSAT Scores')
    ax.set_xlabel('Scores')
    ax.set_xticks([1,2,3,4,5])
    ax.set_ylabel('Frequency')
    return fig

def plot_tenure_distribution (df):
    # Calculating tenure distribution of agents
    tenure_size = df.groupby('Tenure Bucket').size().reset_index(name='count')

    # Plotting the graph
    fig, ax = plt.subplots(figsize=(10,4))
    sns.barplot(tenure_size, x='Tenure Bucket', y='count', palette='magma')
    ax.set_title('Tenure Bucket Distribution')
    ax.set_xlabel('Tenure Bucket')
    ax.set_ylabel('No. of Agents')
    return fig

def plot_channel_name (df):
    # Calculating the numbers of interaction via each channel
    channel = df.groupby('channel_name').size().reset_index(name='count')

    # Plotting the graph
    fig, ax = plt.subplots(figsize=(10,4))
    sns.barplot(channel, x='channel_name', y='count', palette='plasma')
    ax.set_title('Customer Interaction channel distribution')
    ax.set_xlabel('Channel Name')
    ax.set_ylabel('Count')
    return fig

def plot_csat_channel(df):
    # Calculating the average CSAT scores for each channel
    avg_csat_channel = df.groupby('channel_name')['CSAT Score'].mean().reset_index()

    # Plotting the graph
    fig, ax = plt.subplots(figsize=(10,4))
    sns.lineplot(avg_csat_channel, x='channel_name', y='CSAT Score', marker='o', color='midnightblue')
    ax.set_title('Average CSAT Score by Channel')
    ax.set_xlabel('Channel Name')
    ax.set_ylabel('Avg CSAT scores')
    ax.grid(color='y')
    return fig

def plot_category_distribution(df):
    # Calculating the interaction count by category
    category = df.groupby('category').size().reset_index(name='count')

    # Plotting the graph
    fig, ax = plt.subplots(figsize=(10,4))
    sns.barplot(category, x='category', y='count', palette='plasma')
    ax.set_title('Distribution of Customer Interaction by Categories')
    ax.set_xlabel('Categories')
    ax.set_ylabel('Count')
    plt.xticks(rotation=90)
    return fig

def plot_response_distribution(df):
    fig, ax = plt.subplots(figsize=(10,4))
    sns.histplot(df['responsetime_in_mins'], bins=30, kde=True, color='darkgreen')
    ax.set_title('Response Time Distribution')
    ax.set_xlabel('Response Time in mins.')
    ax.set_ylabel('Frequency')
    return fig

def plot_avg_csat_tenure (df):
    # Calculating the average CSAT score for each tenure bucket
    avg_csat_tenure = df.groupby('Tenure Bucket')['CSAT Score'].mean().reset_index()

    # Plotting the graph
    fig, ax  = plt.subplots(figsize=(10,4))
    sns.lineplot(avg_csat_tenure, x='Tenure Bucket', y='CSAT Score', color='darkmagenta', marker='o')
    ax.set_title('Average CSAT Scores across Tenure Buckets')
    ax.set_xlabel('Tenure Bucket')
    ax.set_ylabel('Avg CSAT Scores')
    ax.grid(color='y')
    return fig

def plot_csat_response (df):
    fig, ax = plt.subplots(figsize=(10,4))
    sns.scatterplot(df, x='responsetime_in_mins', y='CSAT Score', color='teal', alpha=0.6)
    ax.set_title('CSAT Scores vs Response Time')
    ax.set_xlabel('Response Time in mins')
    ax.set_ylabel('CSAT Scores')
    ax.set_yticks([1,2,3,4,5])
    return fig

def plot_avg_responsetime_channel(df):
    # Calculating the average response time per channel
    avg_responsetime_channel = df.groupby('channel_name')['responsetime_in_mins'].mean().reset_index()

    # Plotting the graph
    fig, ax = plt.subplots(figsize=(10,4))
    sns.lineplot(avg_responsetime_channel, x='channel_name', y='responsetime_in_mins', color='darkorange')
    ax.grid(color='g')
    ax.set_title('Average Response Time by Channel')
    ax.set_xlabel('Channel Name')
    ax.set_ylabel('Avg Response Time in mins')
    return fig

def plot_csat_category (df):
    # Calculating the average csat scores for each category
    csat_category = df.groupby('category')['CSAT Score'].mean().reset_index()

    # Plotting the graph
    fig, ax = plt.subplots(figsize=(10,4))
    sns.lineplot(csat_category, x='category', y='CSAT Score', color='darkolivegreen', marker='o')
    ax.grid()
    ax.set_title('Average CSAT Scores by Category')
    ax.set_xlabel('Category')
    ax.set_ylabel('Avg CSAT scores')
    plt.xticks(rotation=90)
    return fig

def plot_avg_responsetime_category(df):
    # Calculatinig the average response time for each category
    avg_responsetime_category = df.groupby('category')['responsetime_in_mins'].mean().reset_index()

    # Plotting the graph
    fig, ax = plt.subplots(figsize=(10,4))
    sns.lineplot(avg_responsetime_category, x='category', y='responsetime_in_mins', color='indianred', marker='o')
    ax.set_title('Average Response Time by Category')
    ax.set_xlabel('Categories')
    ax.grid(color='cornflowerblue')
    plt.xticks(rotation=90)
    ax.set_ylabel('Avg Response Time in mins')
    return fig

def plot_agent_interaction_vol (df):
    fig, ax = plt.subplots(figsize=(10,4))
    sns.countplot(df, x='Agent Shift', palette='Set2')
    ax.set_title('Interaction Volume by Agent Shift')
    ax.set_ylabel('Interaction Volume')
    return fig

# Ordinally encoding the 'Tenure Bucket' column
tenure = {'On Job Training': 0,
          '0-30': 1,
          '31-60': 2,
          '61-90': 3,
          '>90': 4}