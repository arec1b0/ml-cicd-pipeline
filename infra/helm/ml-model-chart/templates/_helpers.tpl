{{- define "ml-model-chart.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "ml-model-chart.fullname" -}}
{{- printf "%s" (include "ml-model-chart.name" .) -}}
{{- end -}}
