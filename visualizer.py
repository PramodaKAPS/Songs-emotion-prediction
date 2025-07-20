import matplotlib.pyplot as plt
import seaborn as sns

def create_thayer_plot(predictions_df, output_file):
    if predictions_df.empty:
        print("No data for plot.")
        return
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=predictions_df, x='valence_final', y='arousal_final',
                    hue='valence_final', size='arousal_final',
                    palette='viridis', alpha=0.6, sizes=(20, 200))
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.text(-0.5, 0.5, 'Angry/Stressed', fontsize=10, ha='center')
    plt.text(-0.5, -0.5, 'Sad/Depressed', fontsize=10, ha='center')
    plt.text(0.5, 0.5, 'Happy/Excited', fontsize=10, ha='center')
    plt.text(0.5, -0.5, 'Calm/Peaceful', fontsize=10, ha='center')
    top_songs = predictions_df.nlargest(5, 'arousal_final')
    for _, row in top_songs.iterrows():
        plt.text(row['valence_final'], row['arousal_final'], row['track_name'],
                 fontsize=8, ha='right', va='bottom')
    plt.xlabel('Valence (Negative to Positive)', fontsize=12)
    plt.ylabel('Arousal (Calm to Excited)', fontsize=12)
    plt.title('Thayer\'s Emotion Plane for Spotify Songs (Taylor & Francis 2020)', fontsize=14)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Thayer's plot saved to Google Drive: '{output_file}'")
