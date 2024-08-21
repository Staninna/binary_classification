use std::{
    fs::File,
    io::{BufRead as _, BufReader},
};

struct LogisticRegression {
    weights: Vec<f64>,
    bias: f64,
}

impl LogisticRegression {
    fn new(num_features: usize) -> Self {
        Self {
            weights: vec![0.0; num_features],
            bias: 0.0,
        }
    }

    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    fn predict(&self, inputs: &[f64]) -> f64 {
        let mut z = self.bias;
        for i in 0..inputs.len() {
            z += self.weights[i] * inputs[i];
        }
        Self::sigmoid(z)
    }

    fn train(&mut self, data: Vec<DataPoint>, labels: Vec<Label>, epochs: usize, lr: f64) {
        for epoch in 0..epochs {
            // Update weights and bias using gradient descent
            for i in 0..data.len() {
                let prediction = self.predict(&data[i].features);
                let error = match labels[i] {
                    Label::Malignant => 1.0 - prediction,
                    Label::Benign => 0.0 - prediction,
                };

                // Update weights and bias using gradient descent
                self.bias += lr * error;
                for j in 0..self.weights.len() {
                    self.weights[j] += lr * error * data[i].features[j];
                }
            }

            // Print the loss after each epoch
            let loss = self.compute_loss(&data, &labels);
            println!("Epoch {}: Loss: {:.4}", epoch, loss);
        }
    }

    fn compute_loss(&self, data: &[DataPoint], labels: &[Label]) -> f64 {
        let mut loss = 0.0;
        for i in 0..data.len() {
            let prediction = self.predict(&data[i].features);
            let error = match labels[i] {
                Label::Malignant => 1.0 - prediction,
                Label::Benign => 0.0 - prediction,
            };
            loss += error * error;
        }
        loss
    }
}

#[derive(Clone)]
struct DataPoint {
    id: u32,
    label: Label,
    features: Vec<f64>,
}

impl DataPoint {
    fn len(&self) -> usize {
        self.features.len()
    }
}

#[derive(Clone)]
enum Label {
    Malignant,
    Benign,
}

const DATA_FILE: &str = "breast_cancer_wisconsin_diagnostic/wdbc.data";
const TOTAL_SAMPLES: usize = 569;
const TRAINING_SAMPLES: usize = 400;
const SIMULATION_COUNT: usize = 1000;

fn main() {
    assert!(
        TRAINING_SAMPLES < TOTAL_SAMPLES,
        "Training samples should be less than total samples"
    );

    let mut accuracies = Vec::new();

    for _ in 0..SIMULATION_COUNT {
        // Load data and labels from a file ../breast_cancer_wisconsin_diagnosis/wbcd.data
        let data = load_data(DATA_FILE);
        let labels: Vec<Label> = data.iter().map(|dp| dp.label.clone()).collect();

        // Shuffle the data
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut data: Vec<_> = data.iter().cloned().collect();

        data.shuffle(&mut rng);

        // Split data into training and testing sets
        let test_data = data[TRAINING_SAMPLES..].to_vec();
        let test_labels = labels[TRAINING_SAMPLES..].to_vec();
        let train_data = data[..TRAINING_SAMPLES].to_vec();
        let train_labels = labels[..TRAINING_SAMPLES].to_vec();

        let mut model = LogisticRegression::new(data[0].len());
        model.train(train_data, train_labels, 100, 0.01);

        // Test the model with loaded data
        let mut times_correct = 0;
        for (dp, label) in test_data.iter().zip(test_labels.iter()) {
            let prediction = model.predict(&dp.features);
            let correct = match label {
                Label::Malignant => prediction > 0.5,
                Label::Benign => prediction <= 0.5,
            };
            if correct {
                times_correct += 1;
            }

            println!(
                "ID: {}, Prediction: {:.4}, Label: {} -> {}",
                dp.id,
                prediction,
                match dp.label {
                    Label::Malignant => "Malignant",
                    Label::Benign => "Benign",
                },
                if correct {
                    // green color for correct predictions
                    "\x1b[32mCorrect\x1b[0m"
                } else {
                    // red color for incorrect predictions
                    "\x1b[31mIncorrect\x1b[0m"
                }
            );
        }

        let total_samples = test_data.len();
        let accuracy = times_correct as f64 / total_samples as f64;
        println!("Accuracy: {:.2}%", accuracy * 100.0);
        accuracies.push(accuracy);
    }

    let avg_accuracy: f64 = accuracies.iter().sum::<f64>() / accuracies.len() as f64;

    println!("{:-<20}-+-{:-<22}", "", "");
    println!("{:<20} | {:.4}%", "Average Accuracy", avg_accuracy * 100.0);

    // Print best 10 and worst 10 accuracies
    let mut accuracies = accuracies;
    accuracies.sort_by(|a, b| b.partial_cmp(a).unwrap());

    println!(
        "\n{:<20} | {:<22}",
        "Best accuracies in %", "Worst accuracies in %"
    );
    println!("{:-<20}-+-{:-<22}", "", "");
    for i in 0..10 {
        println!(
            "{:<20.4} | {:<22.4}",
            accuracies[i] * 100.0,
            accuracies[accuracies.len() - 1 - i] * 100.0
        );
    }
}

fn load_data(filename: &str) -> Vec<DataPoint> {
    let mut data = Vec::new();
    let file = File::open(filename).expect("File not found");
    let reader = BufReader::new(file);
    for line in reader.lines() {
        // Parse the line into a DataPoint struct
        let line = line.unwrap();
        let parts: Vec<&str> = line.split(',').collect();
        let id = parts[0].parse().unwrap();
        let label = match parts[1] {
            "M" => Label::Malignant,
            "B" => Label::Benign,
            _ => panic!("Invalid label"),
        };
        let features: Vec<f64> = parts[2..].iter().map(|s| s.parse().unwrap()).collect();
        let dp = DataPoint {
            id,
            label,
            features,
        };
        data.push(dp);
    }

    data
}
