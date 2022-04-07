const FLASK_PORT = 6767;
const fetch_url_data = "http://127.0.0.1:" + FLASK_PORT + "/generate-pdf";


function fn_changeRenderBackgroundColor() {
  let renderBgColor = this.value || "#FFFFFF";

  d3.select("#render-bg-color").attr("value", renderBgColor);
  if (renderBgColor[0] !== "#") {
    renderBgColor = "#" + renderBgColor;
  }
  
  d3.select("#render-area")
    .style("background-color", renderBgColor)
    .style("color", "white");
}


function fn_changeRenderCode() {
  const code = this.value;

  d3.select(this)
    .text(code);

  d3.select("#render-area")
    .text(code)
    .attr("data-processed", null);

  mermaid.init(undefined, document.getElementById("render-area"));
}


function fn_genPdf() {
  const outputUri = d3.select("#output-uri").attr("value");
  const diagram = d3.select("#code-area").text();

  const post_response = {
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
    },
    method: "POST",
    body: JSON.stringify({
      "diagram": diagram,
      "output_uri": outputUri,
    }),
  };

  fetch(fetch_url_data, post_response)
    .then(function (response) { return response.status; })
    .then(function (status_code) { })
    .catch(error => buttonSend.text("Something went wrong."));
}


mermaid.initialize({startOnLoad:true});

d3.select("#render-bg-color")
  .on("input", fn_changeRenderBackgroundColor);

d3.select("#code-area")
  .on("input", fn_changeRenderCode)
  .text(function() { return this.value; } );

d3.select("#output-uri")
  .on("input", function() {
    const outputUri = this.value || "./output.pdf";
    d3.select(this).attr("value", outputUri);
  });

d3.select("#button-gen-pdf")
  .on("click", fn_genPdf);


fn_changeRenderBackgroundColor();
