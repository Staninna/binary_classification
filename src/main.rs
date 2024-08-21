use std::{
    fs::File,
    io::{BufRead as _, BufReader},
    time::Instant,
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

fn main() {
    let start = Instant::now();

    // Load data and labels from a file ../breast_cancer_wisconsin_diagnosis/wbcd.data
    let data = load_data(DATA_FILE);
    let labels: Vec<Label> = data.iter().map(|dp| dp.label.clone()).collect();

    let mut model = LogisticRegression::new(data[0].len());
    model.train(data, labels, 1000, 0.01);

    // Test the model with loaded data
    let test_data = load_data(DATA_FILE);
    let mut times_correct = 0;

    for dp in &test_data {
        let prediction = model.predict(&dp.features);
        let label = match dp.label {
            Label::Malignant => 1.0,
            Label::Benign => 0.0,
        };

        let correct = (prediction - label).abs() < 0.5;
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

    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);
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
