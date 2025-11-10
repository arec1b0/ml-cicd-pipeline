{{- define "ml-model-chart.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "ml-model-chart.fullname" -}}
{{- printf "%s" (include "ml-model-chart.name" .) -}}
{{- end -}}

{{- define "ml-model-chart.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "ml-model-chart.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end -}}
