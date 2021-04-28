use std::fs::File;

use serde::Deserialize;

pub fn read_file_csv<T: for<'de> Deserialize<'de>>(
    file_name: &str,
    size_hint: usize,
) -> Result<Vec<T>, csv::Error> {
    let file_handler = File::open(file_name)?;
    let mut reader = csv::Reader::from_reader(file_handler);

    let mut result_vec = Vec::with_capacity(size_hint);

    for result in reader.deserialize() {
        let v: T = result?;
        result_vec.push(v);
    }
    Ok(result_vec)
}
