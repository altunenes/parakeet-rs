use crate::error::{Error, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Vocabulary parser for vocab.txt format used by TDT models
#[derive(Debug, Clone)]
pub struct Vocabulary {
    pub id_to_token: Vec<String>,
    pub _blank_id: usize,
}

impl Vocabulary {
    /// Load vocabulary from vocab.txt file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| Error::Config(format!("Failed to open vocab file: {}", e)))?;

        let reader = BufReader::new(file);
        let mut id_to_token = Vec::new();
        let mut blank_id = 0;

        for line in reader.lines() {
            let line =
                line.map_err(|e| Error::Config(format!("Failed to read vocab file: {}", e)))?;

            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() == 2 {
                let token = parts[0].to_string();
                let id: usize = parts[1]
                    .parse()
                    .map_err(|e| Error::Config(format!("Invalid token ID in vocab: {}", e)))?;

                if id >= id_to_token.len() {
                    id_to_token.resize(id + 1, String::new());
                }
                id_to_token[id] = token.clone();

                // Track blank token
                if token == "<blk>" || token == "<blank>" {
                    blank_id = id;
                }
            }
        }

        // Default to last token if no blank found
        if blank_id == 0 && !id_to_token.is_empty() {
            blank_id = id_to_token.len() - 1;
        }

        Ok(Self {
            id_to_token,
            _blank_id: blank_id,
        })
    }

    /// Get token by ID
    pub fn id_to_text(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(|s| s.as_str())
    }

    /// Get vocabulary size (number of tokens)
    pub fn size(&self) -> usize {
        self.id_to_token.len()
    }
}
