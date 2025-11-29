import {BadgeQuestionMark, Tag, Pen, Users, Settings, Bookmark, SquarePen, LayoutGrid, LucideIcon } from "lucide-react";
type Submenu = {
  href: string;
  label: string;
  icon: LucideIcon;
  active?: boolean;
};

type Menu = {
  href: string;
  label: string;
  active?: boolean;
  icon: LucideIcon;
  submenus?: Submenu[];
};

type Group = {
  groupLabel: string;
  menus: Menu[];
};

export function getMenuList(pathname: string): Group[] {
  return [
    {
      groupLabel: "",
      menus: [
        {
          href: "/",
          label: "New Chat",
          icon: SquarePen,
          submenus: []
        }
      ]
    },
    {
      groupLabel: "Content",
      menus: [
        {
          href: "",
          label: "Chats",
          icon: Bookmark,
        },  
      ]
    },
    {
      groupLabel: "Other",
      menus: [
        {
          href: "/settings",
          label: "Settings",
          icon: Settings,
          submenus: [
            {
              href: "/settings/account",
              label: "Account",
              icon: Users,
            },
            {
              href: "/settings/change-password",
              label: "Change Password",
              icon: Pen,
            }
          ]
        },
        {
          href: "/help-center",
          label: "Help Center",
          icon: BadgeQuestionMark,
        },
      ]
    }
  ];
}
