
// Complementary error function
// From Numerical Recipes in C 2e p221
const erfc = function(x) {
  const z = Math.abs(x),
    t = 1 / (1 + z / 2),
    r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 +
          t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
          t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
          t * (-0.82215223 + t * 0.17087277)))))))))
  
  return x >= 0 ? r : 2 - r;
};

// Inverse complementary error function
// From Numerical Recipes 3e p265
const ierfc = function(x) {
  if (x >= 2) { return -100; }
  if (x <= 0) { return 100; }

  const xx = (x < 1) ? x : 2 - x;
  const t = Math.sqrt(-2 * Math.log(xx / 2));

  let r = -0.70711 * ((2.30753 + t * 0.27061) /
          (1 + t * (0.99229 + t * 0.04481)) - t);

  for (let j = 0; j < 2; j++) {
    const err = erfc(r) - xx;
    r += err / (1.12837916709551257 * Math.exp(-(r * r)) - r * err);
  }

  return (x < 1) ? r : -r;
};

// Models the normal distribution
const Gaussian = function(mean, variance) {
  if (variance <= 0) {
    throw new Error('Variance must be > 0 (but was ' + variance + ')');
  }
  this.mean = mean;
  this.variance = variance;
  this.standardDeviation = Math.sqrt(variance);
}

// Probability density function
Gaussian.prototype.pdf = function(x) {
  const m = this.standardDeviation * Math.sqrt(2 * Math.PI), 
    e = Math.exp(-Math.pow(x - this.mean, 2) / (2 * this.variance));
  return e / m;
};

// Cumulative density function
Gaussian.prototype.cdf = function(x) {
  return 0.5 * erfc(-(x - this.mean) / (this.standardDeviation * Math.sqrt(2)));
};

// Percent point function
Gaussian.prototype.ppf = function(x) {
  return this.mean - this.standardDeviation * Math.sqrt(2) * ierfc(2 * x);
};

// Product distribution of this and d (scale for constant)
Gaussian.prototype.mul = function(d) {
  if (typeof(d) === "number") {
    return this.scale(d);
  }
  const precision = 1 / this.variance;
  const dprecision = 1 / d.variance;
  return fromPrecisionMean(
      precision + dprecision, 
      precision * this.mean + dprecision * d.mean);
};

// Quotient distribution of this and d (scale for constant)
Gaussian.prototype.div = function(d) {
  if (typeof(d) === "number") {
    return this.scale(1 / d);
  }
  const precision = 1 / this.variance;
  const dprecision = 1 / d.variance;
  return fromPrecisionMean(
      precision - dprecision, 
      precision * this.mean - dprecision * d.mean);
};

// Addition of this and d
Gaussian.prototype.add = function(d) {
  return gaussian(this.mean + d.mean, this.variance + d.variance);
};

// Subtraction of this and d
Gaussian.prototype.sub = function(d) {
  return gaussian(this.mean - d.mean, this.variance + d.variance);
};

// Scale this by constant c
Gaussian.prototype.scale = function(c) {
  return gaussian(this.mean * c, this.variance * c * c);
};

const gaussian = function(mean, variance) {
  return new Gaussian(mean, variance);
};

const fromPrecisionMean = function(precision, precisionmean) {
  return gaussian(precisionmean / precision, 1 / precision);
};

module.exports = gaussian;