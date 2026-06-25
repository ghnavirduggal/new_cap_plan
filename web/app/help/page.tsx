import AppShell from "../_components/AppShell";
import Icon from "../_components/Icon";
import HelpContent from "./help-content";

export default function HelpPage() {
  return (
    <AppShell crumbs="Home" crumbIcon={<Icon name="home" size={16} />}>
      <HelpContent />
    </AppShell>
  );
}
