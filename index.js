//classes: https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
//model: https://tfhub.dev/google/tfjs-model/yamnet/tfjs/1
const modelUrl = "https://tfhub.dev/google/tfjs-model/yamnet/tfjs/1";
const tf = require("@tensorflow/tfjs");
const wavefile = require("wavefile");
const fs = require("fs");
const classes = require("./yamnet_class.json");


const main = async () => {
  await analyzeAudioFile("0/0_239.wav");
  console.log("Finished...");
};

const analyzeAudioFile = async (filename) => {
  console.log("Analyzing audio...");
  const samples = await readWavAudio(filename);
  const scaled = normalizeVector(Array.from(samples));
  predictSound(scaled)
};

const readWavAudio = async (wavFilename) => {
  const wavfiles = fs.readFileSync(wavFilename);
  const wav = new wavefile.WaveFile();
  wav.fromBuffer(wavfiles);
  wav.toSampleRate(16000);
  return wav.getSamples(false, Float32Array);
};

const normalizeVector = (vector) => {
  try {
    const { min, max } = vectorScale(vector[0]);
    const originalScale = [min, max];
    const adjustedScale = [-1, 1];
    return vector[0].map((value) => scaleValue(value, originalScale, adjustedScale));
  } catch (e) {
    const { min, max } = vectorScale(vector);
    const originalScale = [min, max];
    const adjustedScale = [-1, 1];
    return vector.map((value) => scaleValue(value, originalScale, adjustedScale));

  }
  
};

const scaleValue = (value, originalScale, adjustedScale) => {
  const [minO, maxO] = originalScale
  const [minA, maxA] = adjustedScale;
  return ((value - minO) * (maxA - minA)) / (maxO - minO) + minA;
};

const vectorScale = (vector) =>
  vector.reduce(
    (scale, value) => {
      if (value < scale.min) scale.min = value;
      if (value > scale.max) scale.max = value;
      return scale;
    },
    { min: Number.MAX_SAFE_INTEGER, max: Number.MIN_SAFE_INTEGER }
  );


const predictSound = async (wav) => {
  const model = await tf.loadGraphModel(modelUrl, { fromTFHub: true });
  const [scores, embeddings, spectrogram] = model.predict(tf.tensor(wav));

  // embeddings.print(verbose=true);  // shape [N, 1024]
  // spectrogram.print(verbose=true);  // shape [M, 64]
  console.log(scores.mean((axis = 0)).argMax());
  scores
    .mean((axis = 0))
    .argMax()
    .print((verbose = true));
  const predicion = scores.dataSync().slice(0, 521).indexOf(Math.max(...scores.dataSync().slice(0, 521)))

  console.log(`predicion: ${predicion} - ${classes[predicion]}`)
};


main();