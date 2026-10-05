import plotly.graph_objects as go


def plot_microgrid_map(
        mapper,
        title="Microgrid Topology",
        highlighted_nodes=None,
        node_colors=None
):
    """
    Plot IEEE 33-Bus topology on Plotly Mapbox
    """

    edge_lon = []
    edge_lat = []

    for edge in mapper.graph.edges():

        x0 = mapper.bus_coords[edge[0]].x
        y0 = mapper.bus_coords[edge[0]].y

        x1 = mapper.bus_coords[edge[1]].x
        y1 = mapper.bus_coords[edge[1]].y

        edge_lon.extend([x0, x1, None])
        edge_lat.extend([y0, y1, None])

    edge_trace = go.Scattermapbox(
        lon=edge_lon,
        lat=edge_lat,
        mode="lines",
        hoverinfo="none",
        line=dict(
            width=2,
            color="#555"
        )
    )

    node_lon = []
    node_lat = []
    node_text = []
    node_color_list = []

    for node in mapper.graph.nodes():

        node_lon.append(
            mapper.bus_coords[node].x
        )

        node_lat.append(
            mapper.bus_coords[node].y
        )

        node_text.append(
            f"Bus {node}"
        )

        if node_colors and node in node_colors:
            node_color_list.append(
                node_colors[node]
            )

        elif highlighted_nodes and node in highlighted_nodes:
            node_color_list.append("red")

        else:
            node_color_list.append("#1f77b4")

    node_trace = go.Scattermapbox(
        lon=node_lon,
        lat=node_lat,
        mode="markers+text",
        text=[str(n) for n in mapper.graph.nodes()],
        textposition="top right",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=12,
            color=node_color_list
        )
    )

    fig = go.Figure(
        data=[
            edge_trace,
            node_trace
        ]
    )

    fig.update_layout(
        title=title,
        showlegend=False,
        mapbox=dict(
            style="carto-positron",
            center=dict(
                lat=mapper.base_lat,
                lon=mapper.base_lon
            ),
            zoom=12
        ),
        margin=dict(
            l=0,
            r=0,
            t=40,
            b=0
        )
    )

    return fig