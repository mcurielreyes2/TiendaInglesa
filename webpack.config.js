// webpack.config.js

const path = require('path');

module.exports = {
  // 1) The ENTRY point of your entire front-end code:
  entry: "./src/main.js",

  // 2) The OUTPUT bundle
  output: {
    filename: "bundle.js",
    path: __dirname + "/static/dist"
  },

  mode: 'development',
  // or "production" for minification
};