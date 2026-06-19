{{/*
Expand the name of the chart.
*/}}
{{- define "vector-engine.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | cleanSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "vector-engine.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | cleanSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | cleanSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | cleanSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "vector-engine.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | cleanSuffix "-" }}
{{ include "vector-engine.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "vector-engine.selectorLabels" -}}
app.kubernetes.io/name: {{ include "vector-engine.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
