import AppShell from "../_components/AppShell";
import HelpContent from "./help-content";

export default function HelpPage() {
  return (
    <AppShell crumbs="Home" crumbIcon="🏠">
      <HelpContent />
    </AppShell>
  );
}
