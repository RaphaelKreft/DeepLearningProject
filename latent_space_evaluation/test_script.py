import unittest
import numpy as np
from latent_space_evaluation import *
import pandas as pd
from tqdm import tqdm

class TestLatentSpaceEvaluation(unittest.TestCase):
    def setUp(self):
        # Generate random 128-dimensional embeddings
        self.embeddings = np.random.rand(1000, 128)
        self.labels = np.random.randint(0, 5, size=1000)  # Assume 5 classes
        self.n_clusters = 5
        self.output_dir = "./plots"
        self.kmeans_labels = {}

        # Define reducers once for use in both tests
        self.reducers = {
            'PCA': PCAReducer(n_components=2, random_state=42),
            't-SNE': TSNEReducer(
                n_components=2,
                perplexity=30.0,
                learning_rate='auto',
                n_iter=1000,
                random_state=42
            ),
            'UMAP': UMAPReducer(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                random_state=42
            )
        }

    def test_reducers_and_plotting(self):
        # Initialize plotter with verbose arguments
        plotter = EmbeddingPlotter(
            output_dir=self.output_dir,
            palette="tab10",
            fraction=1.0,
            figsize=(10, 8),
            point_size=50,
            alpha=0.7,
            random_state=42
        )
        # Loop over reducers to plot
        for name, reducer in self.reducers.items():
            reduced_embeddings = reducer.fit_transform(self.embeddings)
            plotter.plot(
                embeddings=reduced_embeddings,
                labels=self.labels,
                title=f"{name} Reduction of Embeddings",
                file_name=f"{name.lower()}_plot.png"
            )

    def test_metrics(self):
        # Initialize metrics
        metrics = [
            SilhouetteScoreMetric(),
            DaviesBouldinMetric(),
            AdjustedRandIndexMetric(),
            AdjustedMutualInfoMetric(),
            PurityMetric(),
            AverageEntropyMetric(),
            TrustworthinessMetric(),
            IntraClassCompactnessMetric(),
            InterClassSeparationMetric(),
            CompactnessToSeparationRatio(),
            MutualInformationMetric(),
            UniformityMetric(),
            AlignmentMetric()
        ]

        results = {}

        # Perform K-Means clustering on original embeddings once
        predicted_labels = BaseMetric.cluster_data(self.embeddings, self.n_clusters)

        # Compute metrics for original embeddings
        results["Original"] = {}
        for metric in metrics:
            metric_value = metric.compute(
                embeddings_2d=self.embeddings,
                labels=self.labels,
                n_clusters=self.n_clusters,
                original_embeddings=self.embeddings,
                predicted_labels=predicted_labels
            )
            results["Original"][metric.__class__.__name__] = metric_value

        # Optionally, if you need metrics for reduced embeddings, you can compute them here:
        # for name, reducer in self.reducers.items():
        #     reduced_embeddings = reducer.fit_transform(self.embeddings)
        #     # compute or store metrics if desired

        # Print all metric results
        for reducer_name, metric_results in results.items():
            print(f"\nMetrics for {reducer_name}:")
            for metric_name, value in metric_results.items():
                print(f"{metric_name}: {value}")

def evaluate_embeddings(embeddings, labels, output_dir="./plots"):
    print("[INFO] Starting evaluation of embeddings...")
    reducers = {
        'PCA': PCAReducer(n_components=2, random_state=42),
        't-SNE': TSNEReducer(
            n_components=2,
            perplexity=30.0,
            learning_rate='auto',
            n_iter=1000,
            random_state=42
        ),
        'UMAP': UMAPReducer(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )
    }
    metrics = [
        SilhouetteScoreMetric(),
        DaviesBouldinMetric(),
        AdjustedRandIndexMetric(),
        AdjustedMutualInfoMetric(),
        PurityMetric(),
        AverageEntropyMetric(),
        TrustworthinessMetric(),
        IntraClassCompactnessMetric(),
        InterClassSeparationMetric(),
        CompactnessToSeparationRatio(),
        MutualInformationMetric(),
        UniformityMetric(),
        AlignmentMetric()
    ]
    results = {}

    # Perform K-Means once on original embeddings
    predicted_labels = BaseMetric.cluster_data(embeddings, 5)

    # Compute metrics for original embeddings
    results["Original"] = {}
    for metric in metrics:
        metric_value = metric.compute(
            embeddings_2d=embeddings,
            labels=labels,
            n_clusters=5,
            original_embeddings=embeddings,
            predicted_labels=predicted_labels
        )
        results["Original"][metric.__class__.__name__] = metric_value

    # Loop once over reducers for both transformation & plotting
    plotter = EmbeddingPlotter(
        output_dir=output_dir,
        palette="tab10",
        fraction=1.0,
        figsize=(10, 8),
        point_size=50,
        alpha=0.7,
        random_state=42
    )
    for name, reducer in tqdm(reducers.items(), desc="[INFO] Applying reducers"):
        reduced_embeddings = reducer.fit_transform(embeddings)
        # Optionally compute metrics on reduced embeddings here, if needed
        # ...existing code...
        # Now plot
        print(f"[INFO] Plotting {name} Reduction of Embeddings...")
        plotter.plot(
            embeddings=reduced_embeddings,
            labels=labels,
            title=f"{name} Reduction of Embeddings",
            file_name=f"{name.lower()}_plot.png"
        )

    print("[INFO] Creating DataFrame of metric results...")
    df = pd.DataFrame.from_dict({k: v for k, v in results.items()}, orient='index')
    df.to_csv(f"{output_dir}/metrics.csv")
    print("[INFO] Metrics saved to metrics.csv.")

    # Print all metric results
    for reducer_name, metric_results in results.items():
        print(f"\nMetrics for {reducer_name}:")
        for metric_name, value in metric_results.items():
            print(f"{metric_name}: {value}")
            
def main():
    num_classes = 5
    embeddings = np.random.rand(500, 64)
    labels = np.random.randint(0, num_classes, size=500)  # Assume 5 classes
    output_dir = "./plots"
    evaluate_embeddings(embeddings, labels, output_dir=output_dir)
    print("[INFO] Evaluation complete. Metrics and plots saved to 'results'.")

if __name__ == '__main__':
    main()