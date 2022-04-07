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
  d3.select("#render-area")
    .text(code)
    .attr("data-processed", null);

  mermaid.init(undefined, document.getElementById("render-area"));
}

mermaid.initialize({startOnLoad:true});

fn_changeRenderBackgroundColor();

d3.select("#render-bg-color")
  .on("input", fn_changeRenderBackgroundColor);

d3.select("#code-area")
  .on("input", fn_changeRenderCode);
