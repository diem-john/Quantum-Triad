import numpy as np


# --- PHYSICAL METEOROLOGICAL MODELING ---
def haversine(lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance between two points in km."""
    R = 6371.0 # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def rankine_vortex(v_max, r, r_max=30, x=0.5):
    """Calculates local wind speed at distance r (km) using the Rankine decay model."""
    if r == 0: return 0
    if r <= r_max:
        return v_max * (r / r_max)
    else:
        return v_max * (r_max / r)**x

def vulnerability_curve(local_wind):
    """Fragility curve mapping local wind speed (m/s) to physical failure probability."""
    if local_wind < 15: # Below Tropical Storm threshold
        return 0.02
    elif local_wind < 30: # Tropical Storm to Cat 1
        return 0.02 + 0.04 * (local_wind - 15)
    else: # Cat 2+ Destructive Winds
        return min(0.95, 0.62 + 0.03 * (local_wind - 30))


# The HTML/JS code for the D3.js Interactive Circuit
circuit_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                body { font-family: sans-serif; margin: 0; padding: 10px; color: #333; }
                .controls { display: flex; gap: 20px; margin-bottom: 15px; align-items: center; background: #f8f9fa; padding: 10px; border-radius: 8px;}
                .metrics { margin-bottom: 15px; font-size: 14px; color: #555; }
                svg { background: #ffffff; border: 1px solid #ddd; border-radius: 8px; }
                .wire { stroke: #888; stroke-width: 2px; }
                .gate-box { fill: #e3f2fd; stroke: #1e88e5; stroke-width: 2px; rx: 4px; ry: 4px; }
                .gate-text { font-family: monospace; font-size: 14px; fill: #0d47a1; text-anchor: middle; dominant-baseline: middle; }
                .cnot-line { stroke: #1e88e5; stroke-width: 2px; }
                .cnot-dot { fill: #1e88e5; }
                .cnot-target { fill: none; stroke: #1e88e5; stroke-width: 2px; }
                .measure-box { fill: #f5f5f5; stroke: #666; stroke-width: 2px; rx: 2px; ry: 2px; }
                .divider { stroke: #ccc; stroke-width: 2px; stroke-dasharray: 5,5; }
            </style>
        </head>
        <body>
            <div class="controls">
                <label><b>Qubits:</b> <input type="range" id="q-slider" min="2" max="6" value="3"> <span id="q-val">3</span></label>
                <label><b>Layers:</b> <input type="range" id="l-slider" min="1" max="4" value="2"> <span id="l-val">2</span></label>
            </div>
            <div class="metrics" id="metrics"></div>
            <div style="overflow-x: auto;">
                <svg id="circuit" height="350"></svg>
            </div>

            <script>
                const svg = d3.select("#circuit");
                const wireSpacing = 50;
                const startX = 60;
                const boxSize = 40;
                const stepX = 80;

                function drawCircuit() {
                    const numQubits = parseInt(document.getElementById("q-slider").value);
                    const numLayers = parseInt(document.getElementById("l-slider").value);

                    document.getElementById("q-val").innerText = numQubits;
                    document.getElementById("l-val").innerText = numLayers;

                    const totalParams = numQubits + (numLayers * numQubits * 3);
                    const depth = 1 + (numLayers * 2);
                    document.getElementById("metrics").innerHTML = `<b>Trainable Parameters:</b> ${totalParams} | <b>Gate Depth:</b> ${depth}`;

                    svg.selectAll("*").remove(); 

                    const svgWidth = startX + stepX + (numLayers * stepX * 2) + stepX;
                    svg.attr("width", Math.max(svgWidth, 600));
                    svg.attr("height", numQubits * wireSpacing + 40);

                    // Draw Wires & Labels
                    for (let i = 0; i < numQubits; i++) {
                        let y = 30 + i * wireSpacing;
                        svg.append("line").attr("x1", 20).attr("y1", y).attr("x2", svgWidth - 20).attr("y2", y).attr("class", "wire");
                        svg.append("text").attr("x", 20).attr("y", y).text("|0⟩").attr("dominant-baseline", "middle").style("font-family", "monospace").style("font-weight", "bold");
                    }

                    let currentX = startX;

                    for (let i = 0; i < numQubits; i++) {
                        let y = 30 + i * wireSpacing;
                        svg.append("rect").attr("x", currentX - boxSize/2).attr("y", y - boxSize/2).attr("width", boxSize).attr("height", boxSize).attr("class", "gate-box");
                        svg.append("text").attr("x", currentX).attr("y", y).text("Ry").attr("class", "gate-text");
                    }
                    currentX += stepX;

                    svg.append("line").attr("x1", currentX - stepX/2).attr("y1", 10).attr("x2", currentX - stepX/2).attr("y2", numQubits * wireSpacing + 10).attr("class", "divider");

                    for (let l = 0; l < numLayers; l++) {
                        for (let i = 0; i < numQubits; i++) {
                            let y = 30 + i * wireSpacing;
                            svg.append("rect").attr("x", currentX - boxSize/2).attr("y", y - boxSize/2).attr("width", boxSize).attr("height", boxSize).attr("class", "gate-box");
                            svg.append("text").attr("x", currentX).attr("y", y).text("U(θ)").attr("class", "gate-text");
                        }
                        currentX += stepX;

                        for (let i = 0; i < numQubits; i++) {
                            let controlY = 30 + i * wireSpacing;
                            let targetIdx = (i + 1) % numQubits;
                            let targetY = 30 + targetIdx * wireSpacing;
                            let cnotX = currentX + (i * 10) - ((numQubits*10)/2); 

                            svg.append("line").attr("x1", cnotX).attr("y1", controlY).attr("x2", cnotX).attr("y2", targetY).attr("class", "cnot-line");
                            svg.append("circle").attr("cx", cnotX).attr("cy", controlY).attr("r", 5).attr("class", "cnot-dot"); 
                            svg.append("circle").attr("cx", cnotX).attr("cy", targetY).attr("r", 10).attr("class", "cnot-target"); 
                            svg.append("line").attr("x1", cnotX).attr("y1", targetY - 10).attr("x2", cnotX).attr("y2", targetY + 10).attr("class", "cnot-line"); 
                            svg.append("line").attr("x1", cnotX - 10).attr("y1", targetY).attr("x2", cnotX + 10).attr("y2", targetY).attr("class", "cnot-line"); 
                        }
                        currentX += stepX;
                        svg.append("line").attr("x1", currentX - stepX/2).attr("y1", 10).attr("x2", currentX - stepX/2).attr("y2", numQubits * wireSpacing + 10).attr("class", "divider");
                    }

                    for (let i = 0; i < numQubits; i++) {
                        let y = 30 + i * wireSpacing;
                        svg.append("rect").attr("x", currentX - 15).attr("y", y - 15).attr("width", 30).attr("height", 30).attr("class", "measure-box");
                        svg.append("path").attr("d", `M ${currentX - 8} ${y + 5} Q ${currentX} ${y - 10} ${currentX + 8} ${y + 5}`).attr("fill", "none").attr("stroke", "#666").attr("stroke-width", "2");
                        svg.append("line").attr("x1", currentX).attr("y1", y + 8).attr("x2", currentX + 6).attr("y2", y - 2).attr("stroke", "#666").attr("stroke-width", "2");
                    }
                }

                document.getElementById("q-slider").addEventListener("input", drawCircuit);
                document.getElementById("l-slider").addEventListener("input", drawCircuit);
                drawCircuit();
            </script>
        </body>
        </html>
        """