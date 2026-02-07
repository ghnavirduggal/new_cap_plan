import AppShell from "../_components/AppShell";
import SettingsClient from "./settings-client";

export default function SettingsPage() {
  return (
    <AppShell crumbs="CAP-CONNECT / Settings">
      <SettingsClient />
    </AppShell>
  );
}
