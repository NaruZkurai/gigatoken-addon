// quick: local gigatoken from server tokenizer.json vs server /tokenize
use gigatoken_addon::hftok::LocalHfTokenizer;
fn main(){
    let mut t = LocalHfTokenizer::from_server("192.168.2.64", 6464, 120000).unwrap();
    for s in ["Hello, world!", "The quick brown fox.", "tokenization correctness check 123"] {
        println!("local  {:?} -> {:?}", s, t.encode(s));
    }
}
