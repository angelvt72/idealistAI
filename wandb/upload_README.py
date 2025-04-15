import wandb
import wandb_workspaces.reports.v2 as wr

# Lee el contenido del README.md
with open("./wandb/README_for_w&b_report.md", "r", encoding="utf-8") as f:
    markdown_text = f.read()

# Configuración de tu entidad y proyecto en W&B
entity = "jcriego-prsoria-angelvt"
project = "Understanding-CNNs"

# Crea el reporte en W&B
report = wr.Report(
    entity=entity,
    project=project,
    title="Understanding CNNs",
    description=(
        "This project aims to study and implement models based on Convolutional Neural Networks (CNNs) "
        "by conducting various comparisons that analyze different models and parameters, recording their "
        "impact on prediction quality using Weights and Biases (W&B)."
    ),
)

# Agrega el contenido del README.md como un bloque de Markdown
report.blocks.append(wr.MarkdownBlock(text=markdown_text))

# Guarda (publica) el reporte en W&B
report.save()
